"""Tests for the build tooling: fetch_models.py, make_icons.py, the frozen entry point
and the names that the packaging files, the app and the CI workflows must agree on.

Nothing here touches the network (downloads are faked) or the real system.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import importlib.metadata
import importlib.util
import io
import logging
import ntpath
import os
import platform
import posixpath
import re
import shutil
import struct
import subprocess
import sys
import tomllib
import urllib.error
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from eye_tracker import APP_ID, APP_NAME, __version__

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGING = REPO_ROOT / "packaging"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# A source distribution ships the tests but not the build tooling they check:
# skip instead of failing collection there (the loads below need these files).
if not (REPO_ROOT / "scripts" / "fetch_models.py").is_file() or not PACKAGING.is_dir():
    pytest.skip("build tooling is not part of this source tree", allow_module_level=True)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetch_models = _load(REPO_ROOT / "scripts" / "fetch_models.py", "fetch_models")
make_icons = _load(REPO_ROOT / "scripts" / "make_icons.py", "make_icons")
entry = _load(PACKAGING / "pyinstaller" / "entry.py", "eye_tracker_frozen_entry")


# ============================================================================ fetch_models
def _model(payload: bytes, filename: str = "model.bin") -> Any:
    return fetch_models.Model(
        filename=filename,
        url="https://example.invalid/model.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        license="MIT",
        homepage="https://example.invalid",
        description="test model",
    )


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


FakeUrlopen = tuple[list[Any], list[str]]


@pytest.fixture
def fake_urlopen(monkeypatch: pytest.MonkeyPatch) -> FakeUrlopen:
    """Replace urlopen: ``(outcomes, calls)``; each call pops the next outcome.

    An outcome is the response body (bytes) or an exception to raise.
    """
    outcomes: list[Any] = []
    calls: list[str] = []

    def urlopen(request: Any, timeout: float) -> _FakeResponse:
        calls.append(request.full_url)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeResponse(outcome)

    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(fetch_models.time, "sleep", lambda _s: None)
    return outcomes, calls


def test_pinned_models_match_the_spec() -> None:
    pinned = {m.filename: m.sha256 for m in fetch_models.MODELS}
    assert pinned == {
        "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        "face_detection_yunet_2023mar.onnx": (
            "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
        ),
    }
    assert fetch_models.DEFAULT_DEST == REPO_ROOT / "src" / "eye_tracker" / "vision" / "models"
    assert all(m.url.startswith("https://") for m in fetch_models.MODELS)


def test_model_pins_agree_everywhere(spec_helpers: dict[str, Any]) -> None:
    """fetch_models.py, the backends and the PyInstaller spec pin the same files."""
    from eye_tracker.vision.backends import BACKEND_MODEL_FILES, MODEL_FILES

    installed = dict(pair for model in fetch_models.MODELS for pair in model.installed())
    assert installed == MODEL_FILES
    assert spec_helpers["pinned_models"]() == MODEL_FILES
    assert set().union(*BACKEND_MODEL_FILES.values()) == set(MODEL_FILES)
    # The MediaPipe bundle is only a download source now; it never ships.
    assert "face_landmarker.task" not in MODEL_FILES


def test_verify_statuses(tmp_path: Path) -> None:
    model = _model(b"weights")
    assert fetch_models.verify(model, tmp_path) == "missing"
    (tmp_path / model.filename).write_bytes(b"tampered")
    assert fetch_models.verify(model, tmp_path) == "mismatch"
    (tmp_path / model.filename).write_bytes(b"weights")
    assert fetch_models.verify(model, tmp_path) == "ok"


def test_committed_models_are_valid(capsys: pytest.CaptureFixture[str]) -> None:
    assert fetch_models.main(["--check"]) == 0
    assert capsys.readouterr().out.count("  ok ") == len(fetch_models.MODELS)


def test_check_mode_reports_missing_models_without_downloading(
    tmp_path: Path, fake_urlopen: FakeUrlopen, capsys: pytest.CaptureFixture[str]
) -> None:
    _outcomes, calls = fake_urlopen
    assert fetch_models.main(["--check", "--dest", str(tmp_path)]) == 1
    assert calls == []
    captured = capsys.readouterr()
    assert "missing" in captured.out
    assert f"{len(fetch_models.MODELS)} model(s) missing or invalid" in captured.err


def test_download_writes_verified_file(tmp_path: Path, fake_urlopen: FakeUrlopen) -> None:
    outcomes, calls = fake_urlopen
    model = _model(b"good weights")
    outcomes.append(b"good weights")
    path = fetch_models.download(model, tmp_path)
    assert path == tmp_path / model.filename
    assert path.read_bytes() == b"good weights"
    assert calls == [model.url]
    assert list(tmp_path.glob("*.part")) == []


def test_download_rejects_checksum_mismatch_and_keeps_old_file(
    tmp_path: Path, fake_urlopen: FakeUrlopen
) -> None:
    outcomes, calls = fake_urlopen
    model = _model(b"expected")
    (tmp_path / model.filename).write_bytes(b"previous")
    outcomes.append(b"evil")
    with pytest.raises(fetch_models.DownloadError, match="checksum mismatch"):
        fetch_models.download(model, tmp_path)
    # Never retried, never replaced, no temporary file left behind.
    assert len(calls) == 1
    assert (tmp_path / model.filename).read_bytes() == b"previous"
    assert list(tmp_path.glob("*.part")) == []


def test_download_retries_transient_errors(tmp_path: Path, fake_urlopen: FakeUrlopen) -> None:
    outcomes, calls = fake_urlopen
    model = _model(b"weights")
    outcomes += [urllib.error.URLError("temporary failure"), TimeoutError("slow"), b"weights"]
    assert fetch_models.download(model, tmp_path, attempts=3).read_bytes() == b"weights"
    assert len(calls) == 3


def test_download_gives_up_after_attempts(tmp_path: Path, fake_urlopen: FakeUrlopen) -> None:
    outcomes, calls = fake_urlopen
    outcomes += [urllib.error.URLError("down")] * 2
    with pytest.raises(fetch_models.DownloadError, match="download failed"):
        fetch_models.download(_model(b"x"), tmp_path, attempts=2)
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_download_does_not_retry_client_errors(tmp_path: Path, fake_urlopen: FakeUrlopen) -> None:
    outcomes, calls = fake_urlopen
    model = _model(b"x")
    outcomes.append(urllib.error.HTTPError(model.url, 404, "Not Found", {}, None))  # type: ignore[arg-type]
    with pytest.raises(fetch_models.DownloadError, match="404"):
        fetch_models.download(model, tmp_path)
    assert len(calls) == 1


def test_download_refuses_oversized_responses(
    tmp_path: Path, fake_urlopen: FakeUrlopen, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes, _calls = fake_urlopen
    monkeypatch.setattr(fetch_models, "MAX_BYTES", 10)
    monkeypatch.setattr(fetch_models, "_CHUNK", 4)
    outcomes.append(b"x" * 64)
    with pytest.raises(fetch_models.DownloadError, match="larger than"):
        fetch_models.download(_model(b"x" * 64), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_main_downloads_only_what_is_missing(
    tmp_path: Path, fake_urlopen: FakeUrlopen, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes, calls = fake_urlopen
    present, missing = _model(b"a", "a.bin"), _model(b"b", "b.bin")
    monkeypatch.setattr(fetch_models, "MODELS", (present, missing))
    (tmp_path / "a.bin").write_bytes(b"a")
    outcomes.append(b"b")
    assert fetch_models.main(["--dest", str(tmp_path)]) == 0
    assert calls == [missing.url]
    assert (tmp_path / "b.bin").read_bytes() == b"b"

    # --force downloads everything again.
    outcomes += [b"a", b"b"]
    assert fetch_models.main(["--force", "--dest", str(tmp_path)]) == 0
    assert len(calls) == 3


# ============================================================================== make_icons
def _solid_renderer(color: str) -> Any:
    from PySide6.QtGui import QColor, QImage

    def render(size: int, padded: bool) -> QImage:
        image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(color))
        return image

    return render


def test_qimage_to_pillow_keeps_colour_and_alpha(qapp: Any) -> None:
    from PySide6.QtGui import QColor, QImage

    image = QImage(3, 2, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    image.setPixelColor(0, 0, QColor(255, 0, 0, 255))
    image.setPixelColor(2, 1, QColor(20, 40, 200, 255))
    pil = make_icons._to_pil(image)
    assert pil.mode == "RGBA"
    assert pil.size == (3, 2)
    assert pil.getpixel((0, 0)) == (255, 0, 0, 255)
    assert pil.getpixel((2, 1)) == (20, 40, 200, 255)
    assert pil.getpixel((1, 0))[3] == 0


def test_renderer_size_is_checked(qapp: Any) -> None:
    from PySide6.QtGui import QImage

    def wrong(size: int, padded: bool) -> QImage:
        return QImage(size + 1, size, QImage.Format.Format_ARGB32)

    with pytest.raises(RuntimeError, match="renderer returned"):
        make_icons._render(wrong, 16)


def test_write_icons_produces_every_format(qapp: Any, tmp_path: Path) -> None:
    from PIL import Image

    written = make_icons.write_icons(_solid_renderer("#3366cc"), tmp_path)
    assert sorted(written) == sorted(tmp_path / rel for rel in make_icons.OUTPUTS.values())
    assert make_icons.missing_outputs(tmp_path) == []

    with Image.open(tmp_path / make_icons.OUTPUTS["png"]) as png:
        assert png.size == (1024, 1024)
        assert png.mode == "RGBA"
    with Image.open(tmp_path / make_icons.OUTPUTS["linux"]) as png:
        assert png.size == (512, 512)
    with Image.open(tmp_path / make_icons.OUTPUTS["ico"]) as ico:
        assert set(ico.info["sizes"]) == {(s, s) for s in make_icons.ICO_SIZES}
    with Image.open(tmp_path / make_icons.OUTPUTS["icns"]) as icns:
        assert icns.size == (1024, 1024)


def test_app_icon_renders_with_the_real_renderer(qapp: Any) -> None:
    renderer, _app = make_icons._app_renderer()
    for padded in (False, True):
        image = make_icons._render(renderer, 64, padded=padded)
        assert image.size == (64, 64)
        # The corner is transparent (rounded tile), the centre is opaque (the eye).
        assert image.getpixel((0, 0))[3] == 0
        assert image.getpixel((32, 32))[3] == 255


def test_check_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert make_icons.main(["--check", "--out", str(tmp_path)]) == 1
    assert "missing" in capsys.readouterr().out
    for rel in make_icons.OUTPUTS.values():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    assert make_icons.main(["--check", "--out", str(tmp_path)]) == 0


# ============================================================================= entry point
@pytest.mark.parametrize(
    ("executable", "gui"),
    [
        (r"C:\Users\me\AppData\Local\Programs\Eye Tracker\EyeTracker.exe", True),
        (r"C:\Users\me\AppData\Local\Programs\Eye Tracker\eyetracker.EXE", True),
        ("/Applications/Eye Tracker.app/Contents/MacOS/Eye Tracker", True),
        (r"C:\portable\EyeTracker\eye-tracker-cli.exe", False),
        # The installer's copy of the CLI, found through PATH.
        (r"C:\Users\me\AppData\Local\Programs\Eye Tracker\eye-tracker.exe", False),
        ("/Applications/Eye Tracker.app/Contents/MacOS/eye-tracker-cli", False),
        ("/opt/eye-tracker/eye-tracker", False),
        ("/usr/bin/python3", False),
    ],
)
def test_entry_point_selection(executable: str, gui: bool) -> None:
    assert entry._is_gui_executable(executable) is gui


@pytest.mark.parametrize("flavour", [posixpath, ntpath], ids=["posixpath", "ntpath"])
def test_entry_point_selection_does_not_depend_on_the_os(
    monkeypatch: pytest.MonkeyPatch, flavour: ModuleType
) -> None:
    # os.path is posixpath on the macOS and Linux CI runners, which does not split
    # Windows paths; the helper must give the same answers there (packaging-03).
    monkeypatch.setattr(entry, "os", SimpleNamespace(path=flavour))
    assert entry._is_gui_executable(r"C:\Users\me\Programs\Eye Tracker\EyeTracker.exe")
    assert entry._is_gui_executable(r"C:\Users\me\Programs\Eye Tracker\eyetracker.EXE")
    assert entry._is_gui_executable("/Applications/Eye Tracker.app/Contents/MacOS/Eye Tracker")
    assert not entry._is_gui_executable(r"C:\portable\EyeTracker\eye-tracker-cli.exe")
    assert not entry._is_gui_executable("/opt/eye-tracker/eye-tracker")


@pytest.mark.parametrize(("executable", "expected"), [("EyeTracker.exe", "gui"), ("x-cli", "cli")])
def test_entry_point_dispatch(
    monkeypatch: pytest.MonkeyPatch, executable: str, expected: str
) -> None:
    from eye_tracker import cli

    monkeypatch.setattr(sys, "executable", executable)
    monkeypatch.setattr(cli, "gui_main", lambda: "gui")
    monkeypatch.setattr(cli, "main", lambda: "cli")
    assert entry._run() == expected


# =============================================================== names that must agree
def _spec_text() -> str:
    return (PACKAGING / "pyinstaller" / "eye-tracker.spec").read_text(encoding="utf-8")


def _iss_text() -> str:
    return (PACKAGING / "windows" / "installer.iss").read_text(encoding="utf-8-sig")


def _iss_define(name: str) -> str:
    match = re.search(rf'^#define {name} "([^"]*)"', _iss_text(), re.MULTILINE)
    assert match, name
    return match.group(1)


def test_spec_is_valid_python_and_names_the_executables() -> None:
    source = _spec_text()
    compile(source, "eye-tracker.spec", "exec")
    assert 'GUI_NAME, CLI_NAME, DIST_NAME = "EyeTracker", "eye-tracker-cli", "EyeTracker"' in source
    assert 'GUI_NAME, CLI_NAME, DIST_NAME = None, "eye-tracker", "eye-tracker"' in source
    for key in ("NSCameraUsageDescription", '"LSUIElement": True', '"NSHighResolutionCapable"'):
        assert key in source


def test_gui_executable_names_agree_with_autostart() -> None:
    from eye_tracker.platform import autostart

    gui_names = {"EyeTracker.exe", "Eye Tracker"}
    for name in gui_names:
        assert name in autostart._GUI_EXECUTABLE_NAMES
        assert entry._is_gui_executable(name)
    # On Linux the single "eye-tracker" executable is both, and runs the CLI entry
    # (which starts the tray app when no subcommand is given).
    assert not entry._is_gui_executable("eye-tracker")


def test_installer_matches_the_app() -> None:
    from eye_tracker.platform import autostart

    raw = (PACKAGING / "windows" / "installer.iss").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "installer.iss must keep its UTF-8 BOM"
    assert _iss_define("AppName") == APP_NAME
    assert _iss_define("AppExeName") == "EyeTracker.exe"
    assert _iss_define("CliExeName") == "eye-tracker-cli.exe"
    assert _iss_define("AppUserModelID") == APP_ID
    assert _iss_define("RunValueName") == autostart.RUN_VALUE_NAME
    assert _iss_define("RunKey") == autostart.RUN_KEY
    assert _iss_define("StartupApprovedKey") == autostart.STARTUP_APPROVED_KEY

    text = _iss_text()
    assert "PrivilegesRequired=lowest" in text
    assert r"DefaultDirName={userpf}\{#AppName}" in text
    assert "OutputBaseFilename=EyeTracker-{#AppVersion}-windows-x64-setup" in text
    # The Run value is exactly what autostart.enable(background=True) writes.
    assert 'ValueData: """{app}\\{#AppExeName}"" --background"' in text
    assert autostart.BACKGROUND_FLAG == "--background"


def test_installer_app_id_never_changes() -> None:
    # Upgrades and the upgrade detection in [Code] both depend on it.
    assert _iss_define("AppGuid") == "F2F98F7C-0EEF-44D5-ADDE-173D9D8BA82B"
    text = _iss_text()
    assert "AppId={{{#AppGuid}}" in text
    assert '#define UninstallKey "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{"' in (
        text
    )
    assert '+ AppGuid + "}_is1"' in text


def test_installer_upgrades_keep_the_users_autostart_choice() -> None:
    """packaging-04 / journeys-14: an upgrade must not turn start-at-login back on.

    The behaviour itself is exercised on the disposable release runner (see the
    installer smoke test in release.yml); here the wiring is checked.
    """
    text = _iss_text()
    registry = [
        line for line in text.splitlines() if line.startswith("Root: HKCU") and "startup" in line
    ]
    assert len(registry) == 2
    for line in registry:
        assert line.endswith("Tasks: startup; Check: ShouldWriteAutostart"), line
    code = text[text.index("[Code]") :]
    for fragment in (
        "function ShouldWriteAutostart: Boolean;",
        "else if AutostartWasOn then\n    Result := False",
        "else if WizardSilent or not StartupPageSynced then\n    Result := StartupTaskParam = 1",
        "procedure CurPageChanged(CurPageID: Integer);",
        "WizardSelectTasks('!startup')",
        "IsUpgrade := RegKeyExists(HKCU, '{#UninstallKey}');",
        "AutostartWasOn := IsUpgrade and AutostartEnabledFor(ExpandConstant('{app}'))",
        # Task Manager's "disabled" flag: an odd first byte in StartupApproved.
        "((Ord(Approved[1]) and 1) = 1)",
    ):
        assert fragment in code.replace("\r\n", "\n"), fragment
    # The state is captured before the [Registry] section runs (ssInstall).
    assert code.index("if CurStep = ssInstall then") < code.index("CurStep = ssPostInstall")


def _iss_code() -> str:
    text = _iss_text().replace("\r\n", "\n")
    return text[text.index("[Code]") :]


def _iss_section(name: str) -> list[str]:
    """The non-comment lines of one section of installer.iss."""
    text = _iss_text().replace("\r\n", "\n")
    body = text[text.index(f"\n[{name}]\n") + len(name) + 4 :]
    body = body[: body.index("\n[")]
    return [line for line in body.splitlines() if line and not line.startswith(";")]


def test_installer_puts_the_command_on_path() -> None:
    """r2-docs-01: after a default installation "eye-tracker <command>" works in a new
    terminal, as the documentation writes it; start at sign-in keeps the windowed app."""
    assert _iss_define("CliCommandExeName") == "eye-tracker.exe"
    [task] = [line for line in _iss_section("Tasks") if line.startswith('Name: "addtopath"')]
    assert "unchecked" not in task  # on by default
    assert 'Description: "{cm:AddToPath}"' in task
    assert (
        'Source: "{#BundleDir}\\{#CliExeName}"; DestDir: "{app}"; '
        'DestName: "{#CliCommandExeName}"; Flags: ignoreversion'
    ) in _iss_section("Files")
    assert "ChangesEnvironment=yes" in _iss_section("Setup")
    for language in ("english", "turkish"):
        for message in ("AddToPath", "CommandLineGroup"):
            assert f"{language}.{message}=" in _iss_text()
    code = _iss_code()
    for fragment in (
        # Added (or, when unticked in an upgrade, removed) after the files are in place.
        "if WizardIsTaskSelected('addtopath') then\n      AddAppToPath\n    else\n"
        "      RemoveAppFromPath;",
        # Removed again on uninstall.
        "QuitRunningApp;\n    RemoveAutostartIfOurs;\n    RemoveAppFromPath;",
        # Unexpanded in and out, so %VARIABLES% in the user's PATH survive.
        "RegQueryStringValue(HKCU, EnvironmentKey, 'Path', PathList)",
        "RegWriteExpandStringValue(HKCU, EnvironmentKey, 'Path', PathList + Folder)",
        "RegWriteExpandStringValue(HKCU, EnvironmentKey, 'Path', Kept)",
        "EnvironmentKey = 'Environment';",
    ):
        assert fragment in code, fragment
    # Never added twice.
    add = code[code.index("procedure AddAppToPath;") : code.index("procedure RemoveAppFromPath;")]
    assert "if Found then\n    Exit;" in add
    # The Run value still names the windowed executable.
    [run_value] = [line for line in _iss_section("Registry") if "ValueType: string" in line]
    assert 'ValueData: """{app}\\{#AppExeName}"" --background"' in run_value
    assert "CliCommandExeName" not in run_value


def test_installer_restarts_the_app_after_a_silent_upgrade() -> None:
    """r2-packaging-04: winget's /VERYSILENT upgrade quits the app; it must come back."""
    run = _iss_section("Run")
    assert run == [
        'Filename: "{app}\\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; '
        "Flags: nowait postinstall skipifsilent",
        'Filename: "{app}\\{#AppExeName}"; Parameters: "--background"; '
        "Flags: nowait runasoriginaluser; Check: RelaunchAfterSilentUpgrade",
    ]
    code = _iss_code()
    assert "function RelaunchAfterSilentUpgrade: Boolean;" in code
    assert "Result := WizardSilent and AppWasRunning;" in code
    # Only an instance that acknowledged "ctl quit" counts as running.
    quit_app = code[
        code.index("procedure QuitRunningApp;") : code.index("function PrepareToInstall")
    ]
    assert "(ResultCode <> 0) then\n    Exit;\n  AppWasRunning := True;" in quit_app


def _iscc() -> str | None:
    found = shutil.which("iscc") or shutil.which("ISCC")
    if found:
        return found
    for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES")):
        if base and (Path(base) / "Inno Setup 6" / "ISCC.exe").is_file():
            return str(Path(base) / "Inno Setup 6" / "ISCC.exe")
    return None


@pytest.mark.skipif(sys.platform != "win32" or _iscc() is None, reason="needs Inno Setup 6")
def test_installer_script_compiles(tmp_path: Path) -> None:
    """Compile (never run) the installer against a stub bundle: the [Code] must build."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("EyeTracker.exe", "eye-tracker-cli.exe"):
        (bundle / name).write_bytes(b"stub")
    iscc = _iscc()
    assert iscc is not None
    proc = subprocess.run(
        [
            iscc,
            "/Q",
            "/DAppVersion=0.0.1",
            f"/DBundleDir={bundle}",
            f"/DOutputDir={tmp_path}",
            str(PACKAGING / "windows" / "installer.iss"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "EyeTracker-0.0.1-windows-x64-setup.exe").is_file()


def test_linux_desktop_entry() -> None:
    lines = (PACKAGING / "linux" / "eye-tracker.desktop").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[Desktop Entry]"
    entries = dict(line.split("=", 1) for line in lines[1:] if "=" in line)
    assert entries["Type"] == "Application"
    assert entries["Name"] == APP_NAME
    assert entries["Exec"] == "eye-tracker"
    assert entries["Icon"] == "eye-tracker"
    assert entries["Categories"].endswith(";")


def test_shell_scripts_use_lf_line_endings() -> None:
    for path in (
        PACKAGING / "linux" / "AppRun",
        PACKAGING / "linux" / "build_appimage.sh",
        PACKAGING / "macos" / "make_dmg.sh",
    ):
        data = path.read_bytes()
        assert data.startswith(b"#!"), path
        assert b"\r\n" not in data, f"{path} must use LF line endings"


def test_release_workflow_artifact_names() -> None:
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    for suffix in (
        "windows-x64-setup.exe",
        "windows-x64-portable.zip",
        "macos-arm64.dmg",
        "linux-x86_64.AppImage",
        "linux-x86_64.tar.gz",
    ):
        assert f"EyeTracker-${{{{ env.VERSION }}}}-{suffix}" in text, suffix
    assert "softprops/action-gh-release@" in text
    assert "SHA256SUMS.txt" in text


def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _jobs(text: str) -> dict[str, str]:
    """The text of each job of a workflow, by job id (two-space indented keys)."""
    body = text[text.index("\njobs:\n") :]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


@pytest.mark.parametrize("name", ["ci.yml", "release.yml", "bundle.yml"])
def test_workflow_actions_are_pinned_to_commits(name: str) -> None:
    """packaging-05: a moved tag must not change what runs with release permissions."""
    uses = re.findall(r"^\s*(?:-\s+)?uses:\s*(.+)$", _workflow(name), flags=re.MULTILINE)
    assert uses
    for reference in uses:
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+", reference), reference


def test_dependabot_keeps_actions_and_python_updated() -> None:
    text = (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "package-ecosystem: github-actions" in text
    assert "package-ecosystem: uv" in text
    assert "mediapipe" not in text


def test_release_smoke_tests_use_the_current_backends() -> None:
    from eye_tracker.vision.backends import BACKEND_NAMES

    text = _workflow("release.yml")
    loop = f"for backend in {' '.join(BACKEND_NAMES)}; do"
    assert text.count(loop) == 3
    quoted = ", ".join(f'"{name}"' for name in BACKEND_NAMES)
    assert text.count(f"for backend in ({quoted}):") == 3
    assert "mediapipe" not in text
    assert "opencv" not in text.replace("opencv_", "")


def test_release_builds_pass_the_bundle_privacy_gate() -> None:
    """Every frozen build is scanned before it is packaged or smoke-tested."""
    jobs = _jobs(_workflow("release.yml"))
    for job, bundle in (
        ("build-windows", "dist/EyeTracker"),
        ("build-macos", '"dist/Eye Tracker.app"'),
        ("build-linux", "dist/eye-tracker"),
    ):
        text = jobs[job]
        gate = text.index(
            f"run: uv run --no-sync python scripts/check_privacy.py --bundle {bundle}\n"
        )
        assert text.index("pyinstaller packaging/pyinstaller/eye-tracker.spec") < gate
        packaging_step = {
            "build-windows": "Create the portable ZIP",
            "build-macos": "Sign and create the disk image",
            "build-linux": "Create the AppImage and the tarball",
        }[job]
        assert gate < text.index(packaging_step)


def _apt_packages(text: str) -> list[str]:
    """The packages of the first ``apt-get install`` command in a workflow (or job)."""
    start = text.index("apt-get install -y --no-install-recommends")
    lines = text[start:].splitlines()
    command = []
    for line in lines:
        command.append(line.strip().removesuffix("\\"))
        if not line.rstrip().endswith("\\"):
            break
    return " ".join(command).split()[4:]


_MACOS_MINIMUM = re.compile(r'^  MACOS_MINIMUM: "(\d+\.\d+)"$', re.MULTILINE)


def test_release_checks_the_documented_macos_minimum() -> None:
    """r2-packaging-02: numpy's macosx_14_0 wheel makes the DMG need macOS 14. The
    release fails if LSMinimumSystemVersion stops matching what users are told."""
    text = _workflow("release.yml")
    [minimum] = _MACOS_MINIMUM.findall(text)
    assert minimum == "14.0"
    assert _MACOS_MINIMUM.findall(_workflow("bundle.yml")) == [minimum]
    jobs = _jobs(text)
    macos = jobs["build-macos"]
    assert 'minimum="$(plutil -extract LSMinimumSystemVersion raw "$plist")"' in macos
    assert 'if [[ "$minimum" != "$MACOS_MINIMUM" ]]; then' in macos
    assert macos.index("LSMinimumSystemVersion") < macos.index('"$cli" --version')
    prepare = jobs["prepare"]
    assert 'mac_min = os.environ["MACOS_MINIMUM"].removesuffix(".0")' in prepare
    assert "| macOS {mac_min} or later (Apple silicon) |" in prepare


def test_release_smoke_tests_the_path_entry_and_the_command() -> None:
    windows = _jobs(_workflow("release.yml"))["build-windows"]
    for fragment in (
        '& "$app\\eye-tracker.exe" --version',
        'if ((Get-PathCount) -ne 1) { throw "the installer did not add $app to the user PATH" }',
        'Invoke-Setup "upgrade-no-path" @("/MERGETASKS=!addtopath")',
        'if ((Get-PathCount) -ne 0) { throw "uninstall left $app on the user PATH" }',
        '(Get-Autostart $runKey) -notlike "*\\EyeTracker.exe*"',
    ):
        assert fragment in windows, fragment


def test_bundle_workflow_builds_every_platform_like_the_release() -> None:
    """r2-packaging-01: a bundle that fails the gate shows up on the pull request."""
    text = _workflow("bundle.yml")
    triggers = text[: text.index("\npermissions:")]
    for path in (
        "packaging/**",
        "scripts/check_privacy.py",
        "scripts/fetch_models.py",
        "scripts/make_icons.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/bundle.yml",
    ):
        assert triggers.count(f'      - "{path}"\n') == 2, path  # pull_request and push
    release = _jobs(_workflow("release.yml"))
    for job in ("build-windows", "build-macos", "build-linux"):
        runs_on = re.search(r"^    runs-on: (\S+)$", release[job], re.MULTILINE)
        assert runs_on, job
        assert f"          - os: {runs_on.group(1)}\n" in text, job
    for bundle in ("dist/EyeTracker", "dist/Eye Tracker.app", "dist/eye-tracker"):
        assert f"            bundle: {bundle}\n" in text
    steps = text[text.index("    steps:") :]
    build = steps.index("pyinstaller packaging/pyinstaller/eye-tracker.spec --noconfirm --clean")
    gate = steps.index('run: uv run --no-sync python scripts/check_privacy.py --bundle "$BUNDLE"')
    assert build < gate < steps.index("Smoke-test the frozen CLI")
    assert "upload-artifact" not in text
    assert "contents: write" not in text
    # The same system libraries as the release build, so the same ones get bundled.
    assert _apt_packages(text) == _apt_packages(release["build-linux"])
    assert {"libsm6", "libice6", "libxcb-cursor0"} <= set(_apt_packages(text))


def test_linux_builds_check_that_gtk_and_gio_stay_out() -> None:
    check = "-name 'libgtk-3.so*' -o -name 'libgio-2.0.so*'"
    linux = _jobs(_workflow("release.yml"))["build-linux"]
    assert check in linux
    assert linux.index(check) < linux.index("scripts/check_privacy.py --bundle")
    assert check in _workflow("bundle.yml")


def test_bundle_smoke_test_reads_videos_and_probes_cameras() -> None:
    """The frozen app must still import OpenCV, decode video files with the bundled
    FFmpeg and run the camera backends after the spec leaves libraries out."""
    text = _workflow("bundle.yml")
    smoke = text[text.index("- name: Smoke-test the frozen CLI") :]
    for fragment in (
        "bench --camera packaging/linux/eye-tracker.png",
        '("clip.avi", cv2.CAP_OPENCV_MJPEG, "MJPG")',
        '("clip.mp4", cv2.CAP_FFMPEG, "mp4v")',
        'bench --camera "$clips/$clip"',
        'assert video["modes"]["max"]["analysed"] > 0',
        "doctor --probe-cameras --json > probe.json 2> probe.err",
        '"OpenCV: camera failed to properly initialize!" in errors',
    ):
        assert fragment in smoke, fragment


def test_bug_report_names_the_command_of_every_package() -> None:
    text = (REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(encoding="utf-8")
    for fragment in (
        "`eye-tracker doctor` in a new terminal",
        # Without the PATH entry: spelled so that PowerShell and cmd both run it.
        "`.\\eye-tracker.exe doctor` in `%LOCALAPPDATA%\\Programs\\Eye Tracker`",
        "`.\\eye-tracker-cli.exe doctor`",
        '`"/Applications/Eye Tracker.app/Contents/MacOS/eye-tracker-cli" doctor`',
        "`./EyeTracker-*.AppImage doctor`",
        "`./eye-tracker/eye-tracker doctor`",
        "`uv run eye-tracker doctor`",
    ):
        assert fragment in text, fragment
    # The template's install types are the release packages plus source installs.
    for package in ("setup.exe", "portable ZIP", ".dmg", "AppImage", ".tar.gz", "From source"):
        assert package in text


def test_release_notes_explain_macos_permissions_after_updates() -> None:
    """journeys-15: ad-hoc builds lose the Accessibility grant on every update."""
    prepare = _jobs(_workflow("release.yml"))["prepare"]
    assert "MACOS_STABLE_SIGNATURE: ${{ secrets.MACOS_SIGNING_CERT_P12 != '' }}" in prepare
    assert 'remove\n              Eye Tracker with "−" and add it again.' in prepare
    macos = _jobs(_workflow("release.yml"))["build-macos"]
    # Signing with a stable certificate is optional; without the secret the
    # import step exits early and make_dmg.sh falls back to an ad-hoc signature.
    assert 'if [[ -z "$CERT_P12" ]]; then' in macos
    assert "MACOS_SIGN_IDENTITY: ${{ steps.signing.outputs.identity }}" in macos
    assert "security delete-keychain" in macos


def test_make_dmg_can_sign_with_a_stable_identity() -> None:
    text = (PACKAGING / "macos" / "make_dmg.sh").read_text(encoding="utf-8")
    assert 'IDENTITY="${MACOS_SIGN_IDENTITY:--}"' in text
    assert "--identity) IDENTITY=" in text
    assert 'SIGN_ARGS=(--force --deep --sign "$IDENTITY" --timestamp=none)' in text
    # A certificate-based signature must not leave a build-specific requirement.
    assert '"$REQUIREMENT" == *cdhash*' in text


@pytest.mark.parametrize(
    ("script", "help_range"),
    [("macos/make_dmg.sh", "2,24"), ("linux/build_appimage.sh", "2,27")],
)
def test_script_help_prints_the_whole_header(script: str, help_range: str) -> None:
    lines = (PACKAGING / script).read_text(encoding="utf-8").splitlines()
    assert f"sed -n '{help_range}p'" in "\n".join(lines)
    first, last = (int(n) for n in help_range.split(","))
    header = lines[first - 1 : last]
    assert all(line.startswith("#") for line in header), header
    assert not lines[last].startswith("#")  # the line after the header


_SHA256 = re.compile(r"[0-9a-f]{64}")


def test_appimage_tools_are_pinned() -> None:
    """packaging-06: appimagetool and the embedded runtime are fixed and verified."""
    text = (PACKAGING / "linux" / "build_appimage.sh").read_text(encoding="utf-8")
    assert "continuous" not in text.replace('"continuous" channel', "")
    assert re.search(r'^APPIMAGETOOL_VERSION="\d+\.\d+\.\d+"', text, re.MULTILINE)
    assert re.search(r'^APPIMAGE_RUNTIME_VERSION="\d{8}"', text, re.MULTILINE)
    for tool in ("appimagetool", "runtime"):
        for arch in ("x86_64", "aarch64"):
            match = re.search(rf'^\s+{tool}:{arch}\) echo "([^"]+)" ;;$', text, re.MULTILINE)
            assert match, (tool, arch)
            assert _SHA256.fullmatch(match.group(1)), (tool, arch)
    assert 'TOOL_ARGS=(--no-appstream --runtime-file "$RUNTIME_FILE")' in text
    assert "--allow-unpinned" in text
    assert "refusing to download" in text


_FAKE_CURL = r"""#!/bin/sh
out=""; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    --output) out="$2"; shift 2 ;;
    --retry) shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
echo "$url" >> "$FAKE_CURL_LOG"
case "$url" in
  *appimagetool*) cat "$FAKE_TOOL" > "$out" ;;
  *) printf 'runtime' > "$out" ;;
esac
"""

_FAKE_TOOL = """#!/bin/sh
echo "$@" > "$FAKE_TOOL_LOG"
for last; do :; done
: > "$last"
"""


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bash") is None or shutil.which("sha256sum") is None,
    reason="build_appimage.sh runs on Linux",
)
def test_build_appimage_verifies_its_downloads(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "packaging" / "linux").mkdir(parents=True)
    for name in ("build_appimage.sh", "AppRun", "eye-tracker.desktop", "eye-tracker.png"):
        shutil.copy(PACKAGING / "linux" / name, repo / "packaging" / "linux" / name)
    (repo / "src" / "eye_tracker").mkdir(parents=True)
    (repo / "src" / "eye_tracker" / "__init__.py").write_text('__version__ = "9.9.9"\n')
    (repo / "LICENSE").write_text("MIT\n")
    bundle = repo / "dist" / "eye-tracker"
    (bundle / "_internal").mkdir(parents=True)
    (bundle / "eye-tracker").write_text("#!/bin/sh\n")
    (bundle / "eye-tracker").chmod(0o755)
    (bundle / "_internal" / "libxcb-cursor.so.0").write_bytes(b"x")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(_FAKE_CURL)
    (bin_dir / "curl").chmod(0o755)
    tool = tmp_path / "tool"
    tool.write_text(_FAKE_TOOL)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "ARCH": "x86_64",
        "FAKE_TOOL": str(tool),
        "FAKE_TOOL_LOG": str(tmp_path / "tool.log"),
        "FAKE_CURL_LOG": str(tmp_path / "curl.log"),
    }

    def run(**extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "packaging/linux/build_appimage.sh", "--version", "9.9.9"],
            cwd=repo,
            env={**env, **extra},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    # The real pinned checksums do not match the fake downloads: refused, nothing kept.
    failed = run()
    assert failed.returncode != 0
    assert "checksum mismatch" in failed.stderr
    assert not list((repo / "build" / "tools").glob("*"))

    tool_sha = hashlib.sha256(tool.read_bytes()).hexdigest()
    runtime_sha = hashlib.sha256(b"runtime").hexdigest()
    done = run(APPIMAGETOOL_SHA256=tool_sha, APPIMAGE_RUNTIME_SHA256=runtime_sha)
    assert done.returncode == 0, done.stderr
    downloads = (tmp_path / "curl.log").read_text().split()
    assert downloads[-2:] == [
        "https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage",
        "https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64",
    ]
    arguments = (tmp_path / "tool.log").read_text().split()
    runtime = str(repo / "build" / "tools" / "runtime-20251108-x86_64")
    assert arguments[arguments.index("--runtime-file") + 1] == runtime
    assert (repo / "dist" / "EyeTracker-9.9.9-linux-x86_64.tar.gz").is_file()

    # A custom download without a checksum is refused.
    unpinned = run(APPIMAGETOOL_URL="https://example.invalid/tool.AppImage")
    assert unpinned.returncode != 0
    assert "refusing to download" in unpinned.stderr


def test_pyproject_metadata() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    # packaging-10: the PEP 639 list form of license-files needs hatchling 1.27.
    assert isinstance(project["license-files"], list)
    [hatchling] = [r for r in data["build-system"]["requires"] if r.startswith("hatchling")]
    minimum = re.fullmatch(r"hatchling>=(\d+)\.(\d+)", hatchling)
    assert minimum
    assert (int(minimum.group(1)), int(minimum.group(2))) >= (1, 27)
    # The MediaPipe runtime is gone (packaging-01/08); nothing may pull it back in.
    assert not any("mediapipe" in dep for dep in project["dependencies"])
    assert "mediapipe" not in project["keywords"]
    assert "override-dependencies" not in data.get("tool", {}).get("uv", {})


_DOCS = {"docs/troubleshooting.md", "docs/privacy.md", "docs/platform-support.md"}


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/workflows/release.yml",
    ],
)
def test_documentation_links_point_to_real_pages(path: str) -> None:
    """packaging-12: no links to a docs/ folder listing or to pages that do not exist."""
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    assert "tree/main/docs" not in text
    for target in re.findall(r"eye-tracker/blob/main/([\w./-]+)", text):
        # "docs" alone is the base URL that release.yml formats pages onto.
        assert target in _DOCS | {"CHANGELOG.md", "docs"}, target
    for target in re.findall(r"\{docs\}/([\w.-]+\.md)", text):
        assert f"docs/{target}" in _DOCS, target


def test_model_files_are_binary_in_git() -> None:
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    models = REPO_ROOT / "src" / "eye_tracker" / "vision" / "models"
    suffixes = {p.suffix for p in models.iterdir() if p.is_file() and p.suffix != ".md"}
    assert suffixes
    for suffix in suffixes:
        assert f"*{suffix} binary" in attributes, suffix


def test_sdist_without_build_tooling_skips_these_tests(tmp_path: Path) -> None:
    """packaging-09: the tests that need scripts/ and packaging/ skip in an sdist."""
    tests = tmp_path / "tests"
    tests.mkdir()
    for name in ("test_build_scripts.py", "test_privacy.py"):
        shutil.copy(REPO_ROOT / "tests" / name, tests / name)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", tmp_path],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    # 5 = "no tests collected": every module skipped itself, none errored.
    assert proc.returncode in (0, 5), proc.stdout + proc.stderr
    assert "2 skipped" in proc.stdout
    assert "error" not in proc.stdout.lower()


def test_changelog_has_a_section_for_the_current_version() -> None:
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(
        rf"^## \[{re.escape(__version__)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", text, re.MULTILINE
    ), f"CHANGELOG.md needs a '## [{__version__}] - YYYY-MM-DD' section for the release job"


# ===================================================================== spec helper functions
_SPEC_HELPERS = (
    "runtime_distributions",
    "linux_system_library",
    "hide_foreign_openssl_from_path",
    "_file_name",
    "_without",
    "prune_unreferenced_openssl",
    "prune_orphaned_libraries",
    "macos_minimum_version",
    "pinned_models",
    "verify_models",
    "pinned_licences",
    "verify_licences",
    "_unwanted",
    "_macho_string",
    "_uleb128",
    "_bind_opcodes",
    "_macho_slice",
    "macho_bindings",
    "prune_unused_macos_libraries",
)
# Module-level constants that the helpers above use.
_SPEC_CONSTANTS = (
    "_MH_MAGIC_64",
    "_MH_TWOLEVEL",
    "_LC_SYMTAB",
    "_LC_LOAD_DYLIB",
    "_LC_LAZY_LOAD_DYLIB",
    "_LC_DYLD_INFO",
    "_LC_LOAD_WEAK_DYLIB",
    "_LC_REEXPORT_DYLIB",
    "_LC_DYLD_INFO_ONLY",
    "_LC_LOAD_UPWARD_DYLIB",
    "_LC_DYLD_CHAINED_FIXUPS",
    "_DYLIB_LOAD_COMMANDS",
)


def _constant_names(node: ast.stmt) -> list[str]:
    if not isinstance(node, ast.Assign):
        return []
    return [target.id for target in node.targets if isinstance(target, ast.Name)]


@pytest.fixture(scope="module")
def spec_helpers() -> dict[str, Any]:
    """The spec's helper functions, without running the PyInstaller build it describes."""
    tree = ast.parse(_spec_text())
    body: list[ast.stmt] = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    body += [n for n in tree.body if set(_constant_names(n)) & set(_SPEC_CONSTANTS)]
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _SPEC_HELPERS]
    namespace: dict[str, Any] = {
        "__name__": "eye_tracker_spec_helpers",
        "log": logging.getLogger("eye-tracker.spec.test"),
        "PROJECT": "eye-tracker",
        "PACKAGE": REPO_ROOT / "src" / "eye_tracker",
        "ROOT": REPO_ROOT,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "eye-tracker.spec", "exec"), namespace)
    missing = [name for name in (*_SPEC_HELPERS, *_SPEC_CONSTANTS) if name not in namespace]
    assert not missing, missing
    return namespace


def _canonical(names: list[str]) -> set[str]:
    return {re.sub(r"[-_.]+", "-", name).lower() for name in names}


def test_runtime_distributions_is_the_dependency_closure(spec_helpers: dict[str, Any]) -> None:
    found = _canonical(spec_helpers["runtime_distributions"]())
    assert {"numpy", "opencv-python-headless", "pyside6-essentials", "shiboken6"} <= found
    assert {"psutil", "platformdirs"} <= found
    # Neither the project itself (editable install) nor dev tools or overridden deps.
    assert found.isdisjoint({"eye-tracker", "pytest", "ruff", "mypy", "pyinstaller", "pillow"})
    assert found.isdisjoint({"matplotlib", "sounddevice", "opencv-contrib-python"})
    # The MediaPipe runtime and what it pulled in are gone for good.
    assert found.isdisjoint({"mediapipe", "absl-py", "flatbuffers", "certifi"})


def test_spec_ships_the_opencv_models_and_never_mediapipe() -> None:
    source = _spec_text()
    assert "collect_dynamic_libs" not in source
    assert "mediapipe.tasks" not in source
    # "mediapipe" only appears as an exclusion.
    assert re.findall(r'"mediapipe[^"]*"', source) == ['"mediapipe"']
    excludes = source[source.index("EXCLUDES = [") :]
    assert '\n    "mediapipe",\n' in excludes[: excludes.index("\n]\n")]
    assert 'MODEL_NOTICE = "NOTICE.md"' in source
    # The models' licence texts ship too, checked like the models themselves.
    assert "MODEL_LICENCES = pinned_licences()\n" in source
    assert "verify_licences(MODEL_LICENCES, MODELS_DIR)" in source
    assert "for _model in (*MODEL_FILES, MODEL_NOTICE, *MODEL_LICENCES):" in source


def test_spec_refuses_missing_or_modified_models(
    spec_helpers: dict[str, Any], tmp_path: Path
) -> None:
    good, bad = b"weights", b"tampered"
    pins = {
        "a.tflite": hashlib.sha256(good).hexdigest(),
        "b.onnx": hashlib.sha256(good).hexdigest(),
    }
    (tmp_path / "a.tflite").write_bytes(good)
    (tmp_path / "b.onnx").write_bytes(good)
    spec_helpers["verify_models"](pins, tmp_path)  # all present and intact

    (tmp_path / "b.onnx").write_bytes(bad)
    with pytest.raises(SystemExit, match=r"b\.onnx does not match its pinned SHA-256"):
        spec_helpers["verify_models"](pins, tmp_path)
    (tmp_path / "a.tflite").unlink()
    with pytest.raises(SystemExit, match=r"a\.tflite is missing"):
        spec_helpers["verify_models"](pins, tmp_path)


def test_spec_ships_the_licence_texts_that_fetch_models_pins(
    spec_helpers: dict[str, Any], tmp_path: Path
) -> None:
    """The spec reads the licence texts from fetch_models.py and refuses to build
    without them (Apache-2.0 and MIT both require the text to ship)."""
    pinned = {text.path: text.sha256 for text in fetch_models.LICENCE_TEXTS}
    assert spec_helpers["pinned_licences"]() == pinned
    assert set(pinned) == {"licenses/LICENSE-APACHE-2.0.txt", "licenses/LICENSE-YUNET.txt"}
    models = REPO_ROOT / "src" / "eye_tracker" / "vision" / "models"
    spec_helpers["verify_licences"](pinned, models)  # the committed texts are intact

    good, bad = b"licence text", b"altered text"
    pins = {"licenses/a.txt": hashlib.sha256(good).hexdigest()}
    (tmp_path / "licenses").mkdir()
    with pytest.raises(SystemExit, match=r"licenses/a\.txt is missing\. Restore .* from git"):
        spec_helpers["verify_licences"](pins, tmp_path)
    (tmp_path / "licenses" / "a.txt").write_bytes(bad)
    with pytest.raises(SystemExit, match=r"does not match its pinned SHA-256"):
        spec_helpers["verify_licences"](pins, tmp_path)
    (tmp_path / "licenses" / "a.txt").write_bytes(good)
    spec_helpers["verify_licences"](pins, tmp_path)


class _FakeDist:
    def __init__(self, wheel: str) -> None:
        self._wheel = wheel

    def read_text(self, name: str) -> str | None:
        return self._wheel if name == "WHEEL" else None


@pytest.mark.parametrize(
    ("wheels", "expected"),
    [
        ({"numpy": "Tag: cp312-cp312-win_amd64"}, "12.0"),
        ({"pyside6": "Tag: cp310-abi3-macosx_13_0_universal2"}, "13.0"),
        (
            {
                "pyside6": "Tag: cp310-abi3-macosx_13_0_universal2",
                "numpy": "Tag: cp312-cp312-macosx_14_0_arm64",
                "psutil": "Tag: cp36-abi3-macosx_11_0_arm64",
            },
            "14.0",
        ),
        # A wheel with several tags supports the oldest of them.
        ({"multi": "Tag: py3-none-macosx_15_0_arm64\nTag: py3-none-macosx_11_0_arm64"}, "12.0"),
        ({"old": "Tag: cp312-cp312-macosx_10_13_universal2"}, "12.0"),
    ],
)
def test_macos_minimum_version(
    spec_helpers: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    wheels: dict[str, str],
    expected: str,
) -> None:
    def distribution(name: str) -> _FakeDist:
        if name not in wheels:
            raise importlib.metadata.PackageNotFoundError(name)
        return _FakeDist(wheels[name])

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    names = [*wheels, "not-installed"]
    assert spec_helpers["macos_minimum_version"](names) == expected


@pytest.mark.parametrize(
    ("dest", "unwanted"),
    [
        ("PySide6/plugins/tls/qopensslbackend.dll", True),
        ("PySide6\\plugins\\TLS\\qschannelbackend.dll", True),
        ("PySide6/Qt/plugins/networkinformation/libqnetworkmanager.so", True),
        ("PySide6/plugins/generic/qtuiotouchplugin.dll", True),
        ("PySide6/opengl32sw.dll", True),
        ("PySide6/translations/qtbase_de.qm", True),
        ("PySide6/Qt/translations/qt_tr.qm", True),
        ("PySide6/Qt/plugins/networkaccess/libqnetworkaccessbackend.so", True),
        ("PySide6/Qt/plugins/networkinformation/libqglib.so", True),
        # r2-packaging-05: network servers selectable with QT_QPA_PLATFORM.
        ("PySide6/Qt/plugins/platforms/libqvnc.so", True),
        ("PySide6/Qt/plugins/platforms/libqwebgl.so", True),
        # Full-screen targets without a desktop (and their libinput/udev/gbm stack).
        ("PySide6/Qt/plugins/platforms/libqeglfs.so", True),
        ("PySide6/Qt/plugins/platforms/libqlinuxfb.so", True),
        ("PySide6/Qt/plugins/platforms/libqminimalegl.so", True),
        ("PySide6/Qt/plugins/platforms/libqvkkhrdisplay.so", True),
        ("PySide6/Qt/plugins/egldeviceintegrations/libqeglfs-kms-integration.so", True),
        ("PySide6/Qt/plugins/generic/libqevdevmouseplugin.so", True),
        # r2-packaging-01: the GTK3 theme drags in GTK, Pango, Cairo and GIO.
        ("PySide6/Qt/plugins/platformthemes/libqgtk3.so", True),
        ("PySide6/Qt/plugins/platformthemes/libqxdgdesktopportal.so", False),
        ("PySide6/plugins/platforms/qwindows.dll", False),
        ("PySide6/plugins/platforms/qminimal.dll", False),
        ("PySide6/plugins/platforms/qoffscreen.dll", False),
        ("PySide6/Qt/plugins/platforms/libqxcb.so", False),
        ("PySide6/Qt/plugins/platforms/libqwayland.so", False),
        ("PySide6/Qt/plugins/platforms/libqwayland-egl.so", False),
        ("PySide6/Qt/plugins/platforms/libqoffscreen.so", False),
        ("PySide6/Qt/plugins/platforms/libqminimal.so", False),
        ("PySide6/Qt/plugins/xcbglintegrations/libqxcb-glx-integration.so", False),
        ("PySide6/Qt/plugins/platforminputcontexts/libibusplatforminputcontextplugin.so", False),
        ("PySide6/Qt/plugins/wayland-decoration-client/libbradient.so", False),
        ("PySide6/Qt/plugins/platforms/libqcocoa.dylib", False),
        ("PySide6/plugins/imageformats/qico.dll", False),
        ("PySide6/Qt/lib/libQt6Network.so.6", False),
        ("cv2/cv2.pyd", False),
        ("eye_tracker/vision/models/face_landmarks_detector.tflite", False),
        ("eye_tracker/vision/models/NOTICE.md", False),
    ],
)
def test_unwanted_qt_files(spec_helpers: dict[str, Any], dest: str, unwanted: bool) -> None:
    assert spec_helpers["_unwanted"](dest) is unwanted


def test_spec_removes_every_plugin_the_gate_rejects(spec_helpers: dict[str, Any]) -> None:
    """The bundle gate's network-plugin rule and the spec's removal list agree."""
    check_privacy = _load(REPO_ROOT / "scripts" / "check_privacy.py", "check_privacy_build_tests")
    assert check_privacy.QT_NETWORK_PLUGINS
    for group, pattern, _reason in check_privacy.QT_NETWORK_PLUGINS:
        core = pattern.strip("*") or "backend"
        for dest in (
            f"PySide6/Qt/plugins/{group}/libq{core}.so",
            f"PySide6/plugins/{group}/q{core}.dll",
            f"PySide6/Qt/plugins/{group}/libq{core}.dylib",
        ):
            assert check_privacy.network_plugin(f"_internal/{dest}") is not None, dest
            assert spec_helpers["_unwanted"](dest), dest


def _entries(*items: tuple[str, str]) -> list[tuple[str, str, str]]:
    """TOC entries ``(dest, source, kind)``; the source ends with the dest's file name."""
    return [(dest, f"/src/{dest}", kind) for dest, kind in items]


def _dests(entries: list[tuple[str, str, str]]) -> list[str]:
    return [dest for dest, _src, _kind in entries]


# What ldd reports on Linux: the whole closure of each library, not only DT_NEEDED.
_LINUX_IMPORTS = {
    "libqgtk3.so": {
        "libQt6Gui.so.6",
        "libQt6Core.so.6",
        "libgtk-3.so.0",
        "libgio-2.0.so.0",
        "libglib-2.0.so.0",
        "libc.so.6",
    },
    "libgtk-3.so.0": {"libgio-2.0.so.0", "libglib-2.0.so.0", "libc.so.6"},
    "libgio-2.0.so.0": {"libglib-2.0.so.0", "libc.so.6"},
    "libqxcb.so": {"libQt6XcbQpa.so.6", "libQt6Gui.so.6", "libQt6Core.so.6", "libglib-2.0.so.0"},
    "libQt6XcbQpa.so.6": {"libQt6Gui.so.6", "libQt6Core.so.6", "libglib-2.0.so.0"},
    "libQt6Gui.so.6": {"libQt6Core.so.6", "libglib-2.0.so.0"},
    "libQt6Core.so.6": {"libglib-2.0.so.0"},
    "QtGui.abi3.so": {"libQt6Gui.so.6", "libQt6Core.so.6", "libglib-2.0.so.0"},
}


def test_prune_drops_what_only_removed_plugins_link(
    spec_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """r2-packaging-01: removing the GTK3 theme removes GTK and GIO too, but nothing
    a kept file still links, and nothing loaded at run time."""
    _fake_imports(monkeypatch, _LINUX_IMPORTS)
    removed = _entries(("PySide6/Qt/plugins/platformthemes/libqgtk3.so", "BINARY"))
    kept = [
        *_entries(
            ("PySide6/QtGui.abi3.so", "EXTENSION"),
            ("PySide6/Qt/plugins/platforms/libqxcb.so", "BINARY"),
            ("PySide6/Qt/lib/libQt6XcbQpa.so.6", "BINARY"),
            ("PySide6/Qt/lib/libQt6Gui.so.6", "BINARY"),
            ("PySide6/Qt/lib/libQt6Core.so.6", "BINARY"),
            ("libglib-2.0.so.0", "BINARY"),
            ("libgtk-3.so.0", "BINARY"),
            ("gtk/libgio-2.0.so.0", "BINARY"),
            # Opened with dlopen()/LoadLibrary, so in no import table (like OpenCV's FFmpeg).
            ("cv2/opencv_videoio_ffmpeg500_64.dll", "BINARY"),
            ("eye_tracker/vision/models/NOTICE.md", "DATA"),
        ),
        # PyInstaller's links from the top-level folder to libraries in subfolders.
        ("libQt6Gui.so.6", "PySide6/Qt/lib/libQt6Gui.so.6", "SYMLINK"),
        ("libgio-2.0.so.0", "gtk/libgio-2.0.so.0", "SYMLINK"),
    ]
    pruned = spec_helpers["prune_orphaned_libraries"](kept, removed)
    assert _dests(pruned) == [
        "PySide6/QtGui.abi3.so",
        "PySide6/Qt/plugins/platforms/libqxcb.so",
        "PySide6/Qt/lib/libQt6XcbQpa.so.6",
        "PySide6/Qt/lib/libQt6Gui.so.6",
        "PySide6/Qt/lib/libQt6Core.so.6",
        "libglib-2.0.so.0",
        "cv2/opencv_videoio_ffmpeg500_64.dll",
        "eye_tracker/vision/models/NOTICE.md",
        "libQt6Gui.so.6",
    ]


def test_prune_never_drops_extension_modules(
    spec_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_imports(monkeypatch, {"qtuiotouchplugin.dll": {"QtNetwork.pyd", "Qt6Network.dll"}})
    removed = _entries(("PySide6/plugins/generic/qtuiotouchplugin.dll", "BINARY"))
    kept = _entries(("PySide6/QtNetwork.pyd", "EXTENSION"), ("PySide6/Qt6Network.dll", "BINARY"))
    # Qt6Network.dll is linked by nothing that stays in this toy bundle: it goes;
    # the extension module stays whatever links it.
    assert _dests(spec_helpers["prune_orphaned_libraries"](kept, removed)) == [
        "PySide6/QtNetwork.pyd"
    ]


def test_prune_keeps_everything_when_a_needed_file_is_unreadable(
    spec_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_imports(monkeypatch, {"qopensslbackend.dll": {"libssl-3-x64.dll"}})
    removed = _entries(("PySide6/plugins/tls/qopensslbackend.dll", "BINARY"))
    kept = _entries(("unreadable.dll", "BINARY"), ("libssl-3-x64.dll", "BINARY"))
    assert spec_helpers["prune_orphaned_libraries"](kept, removed) == kept
    # Nothing removed, or nothing it links is bundled: nothing to do either.
    assert spec_helpers["prune_orphaned_libraries"](kept, []) == kept
    _fake_imports(monkeypatch, {"qopensslbackend.dll": {"KERNEL32.dll"}})
    assert spec_helpers["prune_orphaned_libraries"](kept, removed) == kept


def test_without_drops_binaries_and_their_links(spec_helpers: dict[str, Any]) -> None:
    entries = [
        ("PySide6/Qt/lib/libfoo.so.1", "/src/libfoo.so.1", "BINARY"),
        ("libfoo.so.1", "PySide6/Qt/lib/libfoo.so.1", "SYMLINK"),
        ("libbar.so.1", "PySide6/Qt/lib/libbar.so.1", "SYMLINK"),
        ("data/libfoo.so.1", "/src/data/libfoo.so.1", "DATA"),
    ]
    kept = spec_helpers["_without"](entries, {"PySide6\\Qt\\lib\\libfoo.so.1"})
    assert kept == entries[2:]
    assert spec_helpers["_file_name"]("PySide6\\Qt\\lib\\LibFoo.so.1") == "libfoo.so.1"


def _toc(*names: str) -> list[tuple[str, str, str]]:
    return [
        (name, f"/src/{name}", "EXTENSION" if name.endswith(".pyd") else "BINARY") for name in names
    ]


def _fake_imports(monkeypatch: pytest.MonkeyPatch, imports: dict[str, set[str]]) -> None:
    from PyInstaller.depend import bindepend

    def get_imports(src: str, search_paths: object = None) -> set[tuple[str, str | None]]:
        name = src.rsplit("/", 1)[-1]
        if name == "unreadable.dll":
            raise OSError("broken PE header")
        return {(lib, None) for lib in imports.get(name, set())}

    monkeypatch.setattr(bindepend, "get_imports", get_imports)


def test_prune_drops_openssl_nobody_imports(
    spec_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_imports(
        monkeypatch,
        {
            "_hashlib.pyd": {"libcrypto-3-x64.dll", "python312.dll"},
            "libssl-3-x64.dll": {"libcrypto-3-x64.dll"},
            "Qt6Core.dll": {"kernel32.dll"},
        },
    )
    entries = [
        *_toc("_hashlib.pyd", "libcrypto-3-x64.dll", "libssl-3-x64.dll", "Qt6Core.dll"),
        ("certifi/cacert.pem", "/src/cacert.pem", "DATA"),
    ]
    kept = [dest for dest, _src, _kind in spec_helpers["prune_unreferenced_openssl"](entries)]
    assert kept == ["_hashlib.pyd", "libcrypto-3-x64.dll", "Qt6Core.dll", "certifi/cacert.pem"]


def test_prune_keeps_openssl_that_is_imported(
    spec_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A library that needs libssl (as FFmpeg builds may), which in turn needs
    # libcrypto: both stay.
    _fake_imports(
        monkeypatch,
        {"libavformat.so.61": {"libssl.so.3"}, "libssl.so.3": {"libcrypto.so.3"}},
    )
    entries = _toc("libavformat.so.61", "libssl.so.3", "libcrypto.so.3")
    assert spec_helpers["prune_unreferenced_openssl"](entries) == entries


def test_prune_keeps_everything_when_imports_are_unknown(
    spec_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_imports(monkeypatch, {})
    entries = _toc("unreadable.dll", "libssl-3-x64.dll", "libcrypto-3-x64.dll")
    assert spec_helpers["prune_unreferenced_openssl"](entries) == entries


def test_hide_foreign_openssl_from_path(
    spec_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    python_home = tmp_path / "python"
    foreign = tmp_path / "git" / "mingw64" / "bin"
    own = python_home / "DLLs"
    plain = tmp_path / "tools"
    for folder in (foreign, own, plain):
        folder.mkdir(parents=True)
    for folder in (foreign, own):
        (folder / "libssl-3-x64.dll").write_bytes(b"")
    monkeypatch.setattr(sys, "base_prefix", str(python_home))
    monkeypatch.setattr(sys, "prefix", str(python_home))
    path = os.pathsep.join([str(plain), str(foreign), "", str(own), str(tmp_path / "missing")])
    monkeypatch.setenv("PATH", path)

    spec_helpers["hide_foreign_openssl_from_path"]()

    assert os.environ["PATH"].split(os.pathsep) == [
        str(plain),
        "",
        str(own),
        str(tmp_path / "missing"),
    ]


_LDCONFIG = """\
4 libs found in cache `/etc/ld.so.cache'
\tlibxcb-cursor.so.0 (libc6) => /usr/lib/i386-linux-gnu/libxcb-cursor.so.0
\tlibxcb-cursor.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0
\tlibxcb.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libxcb.so.1
"""


def test_linux_system_library_uses_the_linker_cache(
    spec_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=_LDCONFIG, stderr="")

    real_isfile = os.path.isfile
    monkeypatch.setattr(shutil, "which", lambda name: "/sbin/ldconfig")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        os.path, "isfile", lambda path: str(path).startswith("/usr/lib") or real_isfile(path)
    )

    found = spec_helpers["linux_system_library"]("libxcb-cursor.so.0")
    assert found == "/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0"
    assert calls == [["/sbin/ldconfig", "-p"]]
    assert spec_helpers["linux_system_library"]("libnothing.so.9") is None


def test_linux_system_library_falls_back_to_common_folders(
    spec_helpers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_exists, real_glob = os.path.exists, glob.glob
    no_ldconfig = {"/sbin/ldconfig", "/usr/sbin/ldconfig"}
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(os.path, "exists", lambda p: p not in no_ldconfig and real_exists(p))

    def fake_glob(pattern: str) -> list[str]:
        if pattern == "/usr/lib64/libxcb-cursor.so.0":
            return [pattern]
        if pattern.endswith("libxcb-cursor.so.0"):
            return []
        return real_glob(pattern)

    monkeypatch.setattr(glob, "glob", fake_glob)
    found = spec_helpers["linux_system_library"]("libxcb-cursor.so.0")
    assert found == "/usr/lib64/libxcb-cursor.so.0"


# ============================================== macOS: libraries that no bundled binary uses
_LOAD_COMMANDS = {
    "load": 0xC,  # LC_LOAD_DYLIB
    "weak": 0x80000018,  # LC_LOAD_WEAK_DYLIB
    "reexport": 0x8000001F,  # LC_REEXPORT_DYLIB
    "upward": 0x80000023,  # LC_LOAD_UPWARD_DYLIB
}
_LC_LOAD_DYLIB, _LC_LOAD_WEAK_DYLIB = _LOAD_COMMANDS["load"], _LOAD_COMMANDS["weak"]
_DYNAMIC_LOOKUP = 0xFE  # the symbol table's library ordinal for -undefined dynamic_lookup


def _macho_image(
    links: Sequence[tuple[str, str]] = (),
    *,
    imports: Sequence[tuple[str, int]] = (),
    exports: Sequence[str] = (),
    fixups: tuple[int, Sequence[tuple[int, str]]] | None = None,
    compressed_fixup_names: bool = False,
    binds: bytes = b"",
    weak_binds: bytes = b"",
    lazy_binds: bytes = b"",
    twolevel: bool = True,
) -> bytes:
    """A 64-bit little-endian Mach-O dylib, as the spec's binding reader sees one.

    ``links`` are ``(kind, install name)`` load commands (library ordinals 1, 2...),
    ``imports`` symbol-table imports with their library ordinal, ``exports`` the
    symbols the file defines, ``fixups`` ``(import format, [(ordinal, symbol)])``
    for a chained-fixups import table; the bind streams hold dyld-info opcodes.
    """
    commands = b""
    for kind, name in links:
        raw = name.encode() + b"\0"
        raw += b"\0" * (-(24 + len(raw)) % 8)
        commands += struct.pack("<6I", _LOAD_COMMANDS[kind], 24 + len(raw), 24, 2, 0, 0) + raw
    count, tail_size = len(links) + 1, 24  # + LC_SYMTAB
    if fixups is not None:
        count, tail_size = count + 1, tail_size + 16
    if binds or weak_binds or lazy_binds:
        count, tail_size = count + 1, tail_size + 48
    start = 32 + len(commands) + tail_size  # where the data after the load commands begins

    strings, symbols = b"\0", b""
    for name, ordinal in imports:
        symbols += struct.pack("<IBBHQ", len(strings), 0x01, 0, ordinal << 8, 0)  # N_UNDF
        strings += name.encode() + b"\0"
    for name in exports:
        symbols += struct.pack("<IBBHQ", len(strings), 0x0F, 1, 0, 0)  # N_SECT
        strings += name.encode() + b"\0"
    blob = symbols + strings
    tail = struct.pack(
        "<6I", 0x2, 24, start, len(symbols) // 16, start + len(symbols), len(strings)
    )
    if fixups is not None:
        import_format, entries = fixups
        table = names = b""
        for ordinal, symbol in entries:
            if import_format == 3:
                table += struct.pack("<QQ", (ordinal & 0xFFFF) | len(names) << 32, 0)
            else:
                table += struct.pack("<I", (ordinal & 0xFF) | len(names) << 9)
                table += b"\0" * 4 if import_format == 2 else b""
            names += symbol.encode() + b"\0"
        header = struct.pack(
            "<7I", 0, 28, 28, 28 + len(table), len(entries), import_format, compressed_fixup_names
        )
        tail += struct.pack("<4I", 0x80000034, 16, start + len(blob), len(header + table + names))
        blob += header + table + names
    if binds or weak_binds or lazy_binds:
        streams: list[int] = []
        for stream in (binds, weak_binds, lazy_binds):
            streams += [start + len(blob), len(stream)]
            blob += stream
        tail += struct.pack("<12I", 0x80000022, 48, 0, 0, *streams, 0, 0)  # LC_DYLD_INFO_ONLY
    flags = 0x80 if twolevel else 0  # MH_TWOLEVEL
    header = struct.pack(
        "<8I", 0xFEEDFACF, 0x0100000C, 0, 6, count, len(commands) + tail_size, flags, 0
    )
    return header + commands + tail + blob


def _universal(*slices: bytes) -> bytes:
    """A universal (fat) file with ``slices`` at page-aligned offsets."""
    offsets, position = [], 0x1000
    for thin in slices:
        offsets.append(position)
        position += -(-len(thin) // 0x1000) * 0x1000
    data = bytearray(struct.pack(">II", 0xCAFEBABE, len(slices)))
    for offset, thin in zip(offsets, slices, strict=True):
        data += struct.pack(">iiIII", 0x0100000C, 0, offset, len(thin), 12)
    for offset, thin in zip(offsets, slices, strict=True):
        data += b"\0" * (offset - len(data)) + thin
    return bytes(data)


def _bind(ordinal: int, symbol: str) -> bytes:
    """dyld-info opcodes binding ``symbol`` from library ``ordinal`` (<= 0: special)."""
    if ordinal <= 0:
        set_ordinal = bytes([0x30 | (ordinal & 0x0F)])  # SET_DYLIB_SPECIAL_IMM
    elif ordinal < 16:
        set_ordinal = bytes([0x10 | ordinal])  # SET_DYLIB_ORDINAL_IMM
    else:
        set_ordinal = bytes([0x20, ordinal])  # SET_DYLIB_ORDINAL_ULEB
    # SET_SYMBOL_TRAILING_FLAGS_IMM, SET_TYPE_IMM, SET_SEGMENT_AND_OFFSET_ULEB, DO_BIND
    return set_ordinal + b"\x40" + symbol.encode() + b"\0\x51\x72\x10\x90"


def _bindings(spec_helpers: dict[str, Any], data: bytes) -> dict[str, Any]:
    info = spec_helpers["macho_bindings"](data)
    assert info is not None
    return info


def _commands(info: dict[str, Any]) -> list[tuple[int, str]]:
    return [(command, name) for _offset, command, name in info["links"]]


def test_macho_bindings_read_the_symbol_table(spec_helpers: dict[str, Any]) -> None:
    """What ``nm -m`` shows: imports by library ordinal, dynamic lookups, definitions."""
    image = _macho_image(
        [
            ("load", "@rpath/libused.1.dylib"),
            ("load", "@rpath/libX11.6.dylib"),
            ("weak", "/usr/lib/libz.1.dylib"),
        ],
        imports=[("_used", 1), ("_PyLong_FromLong", _DYNAMIC_LOOKUP), ("_hook", 0xFF), ("_z", 3)],
        exports=["_exported"],
    )
    info = _bindings(spec_helpers, image)
    assert _commands(info) == [
        (_LC_LOAD_DYLIB, "@rpath/libused.1.dylib"),
        (_LC_LOAD_DYLIB, "@rpath/libX11.6.dylib"),
        (_LC_LOAD_WEAK_DYLIB, "/usr/lib/libz.1.dylib"),
    ]
    # The offsets are those of the load commands, which is what the spec patches.
    for offset, command, _name in info["links"]:
        assert struct.unpack_from("<I", image, offset) == (command,)
    assert info["bound"] == {"@rpath/libused.1.dylib", "/usr/lib/libz.1.dylib"}
    assert info["lookups"] == {"_PyLong_FromLong"}
    assert info["exports"] == {"_exported"}


@pytest.mark.parametrize("import_format", [1, 2, 3])
def test_macho_bindings_read_chained_fixups(
    spec_helpers: dict[str, Any], import_format: int
) -> None:
    """dyld's own import table, in images built for macOS 12 and later."""
    image = _macho_image(
        [
            ("load", "@rpath/liba.dylib"),
            ("load", "@rpath/libb.dylib"),
            ("load", "@rpath/libc.dylib"),
        ],
        fixups=(
            import_format,
            [(2, "_b"), (-2, "_PyFloat_Type"), (-3, "__ZdlPv"), (0, "_self"), (-1, "_main")],
        ),
    )
    info = _bindings(spec_helpers, image)
    assert info["bound"] == {"@rpath/libb.dylib"}
    assert info["lookups"] == {"_PyFloat_Type", "__ZdlPv"}


def test_macho_bindings_read_classic_bind_opcodes(spec_helpers: dict[str, Any]) -> None:
    binds = (
        _bind(1, "_a")
        + _bind(-2, "_flat")
        # SET_DYLIB_ORDINAL_ULEB 3, SET_SYMBOL, SET_ADDEND_SLEB, ADD_ADDR_ULEB, the other
        # three DO_BIND forms and the two THREADED opcodes.
        + b"\x20\x03\x40_c\0\x60\x7f\x80\x88\x01\xa0\x08\xb1\xc0\x02\x08\xd0\x01\xd1"
        + b"\x00"
    )
    lazy_binds = _bind(4, "_d") + b"\x00" + _bind(4, "_d2") + b"\x00"
    weak_binds = b"\x40__ZdlPv\0\x51\x72\x10\x90\x00"
    image = _macho_image(
        [("load", f"lib{name}") for name in "abcde"],
        binds=binds,
        weak_binds=weak_binds,
        lazy_binds=lazy_binds,
    )
    info = _bindings(spec_helpers, image)
    assert info["bound"] == {"liba", "libc", "libd"}
    # Flat lookups, and C++ weak definitions coalesced by name.
    assert info["lookups"] == {"_flat", "__ZdlPv"}


def test_macho_bindings_count_every_library_of_a_flat_namespace_image(
    spec_helpers: dict[str, Any],
) -> None:
    image = _macho_image([("load", "liba"), ("load", "libb")], imports=[("_x", 0)], twolevel=False)
    info = _bindings(spec_helpers, image)
    assert info["bound"] == {"liba", "libb"}
    assert info["lookups"] == {"_x"}


def test_macho_bindings_read_every_slice_of_a_universal_file(spec_helpers: dict[str, Any]) -> None:
    links = [("load", "@rpath/liba.dylib"), ("load", "@rpath/libb.dylib")]
    fat = _universal(
        _macho_image(links, imports=[("_a", 1)]), _macho_image(links, imports=[("_b", 2)])
    )
    info = _bindings(spec_helpers, fat)
    assert info["bound"] == {"@rpath/liba.dylib", "@rpath/libb.dylib"}
    assert [name for _offset, _command, name in info["links"]] == [
        "@rpath/liba.dylib",
        "@rpath/libb.dylib",
    ] * 2
    for offset, command, _name in info["links"]:
        assert struct.unpack_from("<I", fat, offset) == (command,)


@pytest.mark.parametrize(
    "data",
    [b"", b"MZ\x90\0", b"\x7fELF\x02\x01\x01\0", b"\xca\xfe\xba\xbe\0\0\0\x34", b"#!/bin/sh\n"],
)
def test_macho_bindings_ignore_other_files(spec_helpers: dict[str, Any], data: bytes) -> None:
    assert spec_helpers["macho_bindings"](data) is None


_GOOD_IMAGE = _macho_image([("load", "liba")], imports=[("_a", 1)])


@pytest.mark.parametrize(
    "data",
    [
        _GOOD_IMAGE[:40],  # truncated in its load commands
        b"\xce\xfa\xed\xfe" + _GOOD_IMAGE[4:],  # 32-bit
        b"\xfe\xed\xfa\xcf" + _GOOD_IMAGE[4:],  # big-endian
        _macho_image([("load", "liba")], imports=[("_a", 2)]),  # no library 2
        _macho_image([("load", "liba")], fixups=(4, [(1, "_a")])),  # unknown import format
        _macho_image([("load", "liba")], fixups=(1, [(1, "_a")]), compressed_fixup_names=True),
        _macho_image([("load", "liba")], binds=b"\xe0"),  # unknown bind opcode
        _macho_image([("load", "liba")], binds=b"\x40_unterminated"),
    ],
)
def test_macho_bindings_refuse_what_they_cannot_read(
    spec_helpers: dict[str, Any], data: bytes
) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - every reason is a ValueError
        spec_helpers["macho_bindings"](data)


def _write_toc(tmp_path: Path, files: dict[str, tuple[bytes, str]]) -> list[tuple[str, str, str]]:
    """TOC entries for ``{dest: (contents, kind)}``, the files written below ``tmp_path``."""
    entries = []
    for dest, (data, kind) in files.items():
        path = tmp_path / "site" / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append((dest, str(path), kind))
    return entries


_DYLIBS = "cv2/.dylibs/"


def _opencv_like_wheel() -> dict[str, tuple[bytes, str]]:
    """The shape of OpenCV 5.0's macOS wheel, as measured on the runner with ``nm -m``.

    FFmpeg's libraries link libX11 but bind nothing to it (libavdevice binds libxcb
    for its X11 grab device), libsrt links libssl but only uses libcrypto, and
    libjxl links libhwy without using it.
    """
    here = "@loader_path/"
    return {
        "cv2/cv2.abi3.so": (
            _macho_image(
                [
                    ("load", "@loader_path/.dylibs/libavformat.61.dylib"),
                    ("load", "@loader_path/.dylibs/libavdevice.61.dylib"),
                ],
                imports=[
                    ("_avformat_open_input", 1),
                    ("_avdevice_register_all", 2),
                    ("_PyLong_FromLong", _DYNAMIC_LOOKUP),
                ],
            ),
            "EXTENSION",
        ),
        _DYLIBS + "libavformat.61.dylib": (
            _macho_image(
                [
                    ("load", here + "libX11.6.dylib"),
                    ("load", here + "libsrt.1.5.dylib"),
                    ("load", here + "libjxl.0.11.dylib"),
                ],
                imports=[("_srt_startup", 2), ("_JxlDecoderCreate", 3)],
                exports=["_avformat_open_input"],
            ),
            "BINARY",
        ),
        _DYLIBS + "libavdevice.61.dylib": (
            _macho_image(
                [
                    ("load", here + "libavformat.61.dylib"),
                    ("load", here + "libxcb.1.dylib"),
                    ("load", here + "libX11.6.dylib"),
                ],
                imports=[("_avformat_open_input", 1), ("_xcb_connect", 2)],
                exports=["_avdevice_register_all"],
            ),
            "BINARY",
        ),
        _DYLIBS + "libX11.6.dylib": (
            _macho_image(
                [("load", here + "libxcb.1.dylib")],
                imports=[("_xcb_connect", 1)],
                exports=["_XOpenDisplay"],
            ),
            "BINARY",
        ),
        _DYLIBS + "libxcb.1.dylib": (_macho_image(exports=["_xcb_connect"]), "BINARY"),
        _DYLIBS + "libsrt.1.5.dylib": (
            _macho_image(
                [("load", here + "libssl.3.dylib"), ("load", here + "libcrypto.3.dylib")],
                imports=[("_EVP_CIPHER_CTX_new", 2)],
                exports=["_srt_startup"],
            ),
            "BINARY",
        ),
        _DYLIBS + "libssl.3.dylib": (
            _macho_image(
                [("load", here + "libcrypto.3.dylib")],
                imports=[("_EVP_MD_CTX_new", 1)],
                exports=["_SSL_new"],
            ),
            "BINARY",
        ),
        _DYLIBS + "libcrypto.3.dylib": (
            _macho_image(exports=["_EVP_CIPHER_CTX_new", "_EVP_MD_CTX_new"]),
            "BINARY",
        ),
        _DYLIBS + "libjxl.0.11.dylib": (
            _macho_image([("load", here + "libhwy.1.dylib")], exports=["_JxlDecoderCreate"]),
            "BINARY",
        ),
        _DYLIBS + "libhwy.1.dylib": (_macho_image(exports=["__ZN3hwy5AbortEv"]), "BINARY"),
        # Loaded with dlopen(), so linked by nothing: it stays.
        "PySide6/Qt/plugins/platforms/libqcocoa.dylib": (
            _macho_image(
                [("load", "@rpath/QtGui.framework/Versions/A/QtGui")], imports=[("_qt_gui", 1)]
            ),
            "BINARY",
        ),
        "PySide6/Qt/lib/QtGui.framework/Versions/A/QtGui": (
            _macho_image(exports=["_qt_gui"]),
            "BINARY",
        ),
        # Not a Mach-O file: ignored.
        "numpy/odd.so": (b"\x7fELF\x02\x01\x01\0" + b"\0" * 56, "BINARY"),
    }


def test_prune_leaves_out_libraries_that_nothing_binds_to(
    spec_helpers: dict[str, Any], tmp_path: Path
) -> None:
    """libX11, libssl and libhwy are linked by OpenCV's FFmpeg but used by nothing;
    their links become weak in copies of the libraries that link them."""
    files = _opencv_like_wheel()
    entries = [
        *_write_toc(tmp_path, files),
        # PyInstaller's links from the top-level folder, and a data file.
        ("libX11.6.dylib", _DYLIBS + "libX11.6.dylib", "SYMLINK"),
        ("libxcb.1.dylib", _DYLIBS + "libxcb.1.dylib", "SYMLINK"),
        ("cv2/config.py", str(tmp_path / "config.py"), "DATA"),
    ]
    scratch = tmp_path / "scratch"
    pruned = spec_helpers["prune_unused_macos_libraries"](entries, scratch)

    gone = {_DYLIBS + "libX11.6.dylib", _DYLIBS + "libssl.3.dylib", _DYLIBS + "libhwy.1.dylib"}
    assert [dest for dest, _src, _kind in pruned] == [
        dest for dest, _src, _kind in entries if dest not in gone and dest != "libX11.6.dylib"
    ]
    weakened = {
        _DYLIBS + "libavformat.61.dylib": ["@loader_path/libX11.6.dylib"],
        _DYLIBS + "libavdevice.61.dylib": ["@loader_path/libX11.6.dylib"],
        _DYLIBS + "libsrt.1.5.dylib": ["@loader_path/libssl.3.dylib"],
        _DYLIBS + "libjxl.0.11.dylib": ["@loader_path/libhwy.1.dylib"],
    }
    sources = {dest: Path(src) for dest, src, _kind in pruned}
    for dest, (original, _kind) in files.items():
        assert (tmp_path / "site" / dest).read_bytes() == original  # the wheel is untouched
        if dest in gone:
            continue
        if dest not in weakened:
            assert sources[dest] == tmp_path / "site" / dest, dest
            continue
        assert sources[dest] == scratch / dest
        expected = bytearray(original)
        for offset, _command, name in _bindings(spec_helpers, original)["links"]:
            if name in weakened[dest]:
                struct.pack_into("<I", expected, offset, _LC_LOAD_WEAK_DYLIB)
        # Only the type of those load commands changes: library ordinals stay as they are.
        assert sources[dest].read_bytes() == bytes(expected), dest


def test_prune_keeps_libraries_whose_symbols_are_found_by_name(
    spec_helpers: dict[str, Any], tmp_path: Path
) -> None:
    """Python extension modules find the Python API by name (-undefined dynamic_lookup)
    and C++ weak definitions are coalesced by name, so a library that defines such a
    symbol stays even when no link binds anything to it."""
    entries = _write_toc(
        tmp_path,
        {
            "ext.cpython-312-darwin.so": (
                _macho_image(
                    [("load", "@rpath/libpy.dylib"), ("load", "@rpath/libcxx.dylib")],
                    imports=[("_PyLong_FromLong", _DYNAMIC_LOOKUP)],
                    fixups=(1, [(-3, "__ZdlPv")]),
                ),
                "EXTENSION",
            ),
            "libpy.dylib": (_macho_image(exports=["_PyLong_FromLong"]), "BINARY"),
            "libcxx.dylib": (_macho_image(exports=["__ZdlPv"]), "BINARY"),
        },
    )
    assert spec_helpers["prune_unused_macos_libraries"](entries, tmp_path / "scratch") == entries


@pytest.mark.parametrize("kind", ["reexport", "upward"])
def test_prune_keeps_reexported_and_upward_linked_libraries(
    spec_helpers: dict[str, Any], tmp_path: Path, kind: str
) -> None:
    entries = _write_toc(
        tmp_path,
        {
            "ext.so": (
                _macho_image([("load", "@rpath/libumbrella.dylib")], imports=[("_inner", 1)]),
                "EXTENSION",
            ),
            "libumbrella.dylib": (_macho_image([(kind, "@rpath/libinner.dylib")]), "BINARY"),
            "libinner.dylib": (_macho_image(exports=["_inner"]), "BINARY"),
        },
    )
    assert spec_helpers["prune_unused_macos_libraries"](entries, tmp_path / "scratch") == entries


def test_prune_also_drops_what_only_unused_libraries_link(
    spec_helpers: dict[str, Any], tmp_path: Path
) -> None:
    entries = _write_toc(
        tmp_path,
        {
            "ext.so": (
                _macho_image([("load", "@rpath/liba.dylib"), ("weak", "@rpath/libw.dylib")]),
                "EXTENSION",
            ),
            "liba.dylib": (
                _macho_image([("load", "@rpath/libb.dylib")], imports=[("_b", 1)]),
                "BINARY",
            ),
            "libb.dylib": (_macho_image(exports=["_b"]), "BINARY"),
            "libw.dylib": (_macho_image(exports=["_w"]), "BINARY"),
        },
    )
    pruned = spec_helpers["prune_unused_macos_libraries"](entries, tmp_path / "scratch")
    # Extension modules always stay; libb was only needed by liba.
    assert [dest for dest, _src, _kind in pruned] == ["ext.so"]
    info = _bindings(spec_helpers, Path(pruned[0][1]).read_bytes())
    assert _commands(info) == [
        (_LC_LOAD_WEAK_DYLIB, "@rpath/liba.dylib"),
        (_LC_LOAD_WEAK_DYLIB, "@rpath/libw.dylib"),
    ]


def test_prune_weakens_the_links_of_every_slice(
    spec_helpers: dict[str, Any], tmp_path: Path
) -> None:
    thin = _macho_image(
        [("load", "@rpath/libX11.6.dylib"), ("load", "@rpath/libxcb.1.dylib")],
        imports=[("_xcb_connect", 2)],
    )
    entries = _write_toc(
        tmp_path,
        {
            "ext.so": (_universal(thin, thin), "EXTENSION"),
            "libX11.6.dylib": (_macho_image(exports=["_XOpenDisplay"]), "BINARY"),
            "libxcb.1.dylib": (_macho_image(exports=["_xcb_connect"]), "BINARY"),
        },
    )
    pruned = spec_helpers["prune_unused_macos_libraries"](entries, tmp_path / "scratch")
    assert [dest for dest, _src, _kind in pruned] == ["ext.so", "libxcb.1.dylib"]
    info = _bindings(spec_helpers, Path(pruned[0][1]).read_bytes())
    assert (
        _commands(info)
        == [
            (_LC_LOAD_WEAK_DYLIB, "@rpath/libX11.6.dylib"),
            (_LC_LOAD_DYLIB, "@rpath/libxcb.1.dylib"),
        ]
        * 2
    )


def test_prune_keeps_everything_when_a_binary_cannot_be_read(
    spec_helpers: dict[str, Any], tmp_path: Path
) -> None:
    files = _opencv_like_wheel()
    files["broken.dylib"] = (_GOOD_IMAGE[:40], "BINARY")
    entries = _write_toc(tmp_path, files)
    scratch = tmp_path / "scratch"
    assert spec_helpers["prune_unused_macos_libraries"](entries, scratch) == entries
    assert not scratch.exists()


def test_spec_leaves_out_unused_libraries_on_macos_only() -> None:
    source = _spec_text()
    call = source.index("a.binaries = prune_unused_macos_libraries(")
    assert source.rindex("\nif IS_MACOS:\n", 0, call) > source.index(
        "a.binaries = prune_unreferenced_openssl("
    )
