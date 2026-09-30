"""Tests for ``eye_tracker.platform.autostart`` on all three OS code paths.

The OS is selected by patching ``autostart._system``; the registry helpers are
replaced by an in-memory dict and the LaunchAgent / .desktop locations point
into ``tmp_path``. An autouse fixture makes any accidental use of the real
registry helpers fail, so the real ``HKCU\\...\\Run`` key is never touched.
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from eye_tracker import APP_ID, APP_NAME, APP_SLUG, paths
from eye_tracker.platform import autostart

REAL_LINUX_DESKTOP_PATH = autostart._linux_desktop_path
REAL_MAC_PLIST_PATH = autostart._mac_plist_path


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect every location into tmp_path and forbid the real registry."""

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the real registry must not be used in tests")

    monkeypatch.setattr(autostart, "_reg_read", forbidden)
    monkeypatch.setattr(autostart, "_reg_write_str", forbidden)
    monkeypatch.setattr(autostart, "_reg_delete", forbidden)
    monkeypatch.setattr(
        autostart, "_mac_plist_path", lambda: tmp_path / "LaunchAgents" / f"{APP_ID}.plist"
    )
    monkeypatch.setattr(
        autostart, "_linux_desktop_path", lambda: tmp_path / "autostart" / f"{APP_SLUG}.desktop"
    )
    monkeypatch.delenv("APPIMAGE", raising=False)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], Any]:
    """In-memory HKCU: ``{(key_path, value_name): data}``."""
    store: dict[tuple[str, str], Any] = {}

    def read(key_path: str, name: str) -> Any:
        return store.get((key_path, name))

    def write(key_path: str, name: str, value: str) -> None:
        assert isinstance(value, str)
        store[(key_path, name)] = value

    def delete(key_path: str, name: str) -> None:
        store.pop((key_path, name), None)

    monkeypatch.setattr(autostart, "_reg_read", read)
    monkeypatch.setattr(autostart, "_reg_write_str", write)
    monkeypatch.setattr(autostart, "_reg_delete", delete)
    return store


def use_system(monkeypatch: pytest.MonkeyPatch, system: str) -> None:
    monkeypatch.setattr(autostart, "_system", lambda: system)


@pytest.fixture
def source_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fake interpreter directory with python.exe and pythonw.exe."""
    bin_dir = tmp_path / "venv" / "Scripts"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python.exe"
    python.write_text("")
    (bin_dir / "pythonw.exe").write_text("")
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    return python


@pytest.fixture
def frozen_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fake frozen install directory ("C:/Program Files/Eye Tracker"-like)."""
    app_dir = tmp_path / "Program Files" / "Eye Tracker"
    app_dir.mkdir(parents=True)
    gui = app_dir / "EyeTracker.exe"
    gui.write_text("")
    (app_dir / "eye-tracker-cli.exe").write_text("")
    monkeypatch.setattr(sys, "executable", str(gui))
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    return gui


def desktop_exec_split(value: str) -> list[str]:
    """Reference parser for an ``Exec`` value (Desktop Entry spec 1.5)."""
    # 1. key-file string unescaping
    unescaped: list[str] = []
    it = iter(value)
    for ch in it:
        if ch == "\\":
            nxt = next(it)
            unescaped.append({"s": " ", "n": "\n", "t": "\t", "r": "\r", "\\": "\\"}[nxt])
        else:
            unescaped.append(ch)
    text = "".join(unescaped)
    # 2. argument splitting with double-quote rules, 3. %% -> %
    args: list[str] = []
    current: list[str] = []
    in_quotes = False
    started = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_quotes:
            if ch == "\\" and i + 1 < len(text) and text[i + 1] in '"`$\\':
                current.append(text[i + 1])
                i += 1
            elif ch == '"':
                in_quotes = False
            else:
                current.append(ch)
        elif ch == '"':
            in_quotes = True
            started = True
        elif ch == " ":
            if started or current:
                args.append("".join(current))
                current, started = [], False
        else:
            current.append(ch)
        i += 1
    if started or current:
        args.append("".join(current))
    return [a.replace("%%", "%") for a in args]


# ------------------------------------------------------------ launch command
class TestLaunchCommand:
    def test_source_on_windows_prefers_pythonw(
        self, monkeypatch: pytest.MonkeyPatch, source_python: Path
    ) -> None:
        use_system(monkeypatch, "windows")
        pythonw = str(source_python.with_name("pythonw.exe"))
        assert autostart.launch_command() == [pythonw, "-m", "eye_tracker", "--background"]
        assert autostart.launch_command(background=False) == [pythonw, "-m", "eye_tracker"]

    def test_source_on_windows_without_pythonw(
        self, monkeypatch: pytest.MonkeyPatch, source_python: Path
    ) -> None:
        use_system(monkeypatch, "windows")
        source_python.with_name("pythonw.exe").unlink()
        assert autostart.launch_command()[0] == str(source_python)

    @pytest.mark.parametrize("system", ["macos", "linux"])
    def test_source_elsewhere_uses_sys_executable(
        self, monkeypatch: pytest.MonkeyPatch, source_python: Path, system: str
    ) -> None:
        use_system(monkeypatch, system)
        assert autostart.launch_command() == [
            str(source_python),
            "-m",
            "eye_tracker",
            autostart.BACKGROUND_FLAG,
        ]

    def test_frozen(self, monkeypatch: pytest.MonkeyPatch, frozen_app: Path) -> None:
        use_system(monkeypatch, "windows")
        assert autostart.launch_command() == [str(frozen_app), "--background"]
        assert autostart.launch_command(background=False) == [str(frozen_app)]

    def test_frozen_cli_registers_the_windowed_executable(
        self, monkeypatch: pytest.MonkeyPatch, frozen_app: Path
    ) -> None:
        use_system(monkeypatch, "windows")
        monkeypatch.setattr(sys, "executable", str(frozen_app.with_name("eye-tracker-cli.exe")))
        assert autostart.launch_command()[0] == str(frozen_app)

    def test_frozen_cli_without_gui_sibling_keeps_itself(
        self, monkeypatch: pytest.MonkeyPatch, frozen_app: Path
    ) -> None:
        use_system(monkeypatch, "windows")
        cli = frozen_app.with_name("eye-tracker-cli.exe")
        frozen_app.unlink()
        monkeypatch.setattr(sys, "executable", str(cli))
        assert autostart.launch_command()[0] == str(cli)

    def test_appimage(
        self, monkeypatch: pytest.MonkeyPatch, frozen_app: Path, tmp_path: Path
    ) -> None:
        use_system(monkeypatch, "linux")
        image = tmp_path / "Eye Tracker-x86_64.AppImage"
        image.write_text("")
        monkeypatch.setenv("APPIMAGE", str(image))
        assert autostart.launch_command() == [str(image), "--background"]

    def test_missing_appimage_file_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, frozen_app: Path, tmp_path: Path
    ) -> None:
        use_system(monkeypatch, "linux")
        monkeypatch.setenv("APPIMAGE", str(tmp_path / "gone.AppImage"))
        assert autostart.launch_command()[0] == str(frozen_app)

    def test_appimage_env_ignored_for_source_runs(
        self, monkeypatch: pytest.MonkeyPatch, source_python: Path, tmp_path: Path
    ) -> None:
        use_system(monkeypatch, "linux")
        image = tmp_path / "Terminal.AppImage"
        image.write_text("")
        monkeypatch.setenv("APPIMAGE", str(image))
        assert autostart.launch_command()[1:3] == ["-m", "eye_tracker"]


# -------------------------------------------------------------------- Windows
@pytest.mark.usefixtures("frozen_app")
class TestWindows:
    @pytest.fixture(autouse=True)
    def _windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_system(monkeypatch, "windows")

    def test_enable_writes_quoted_run_value(
        self, registry: dict[tuple[str, str], Any], frozen_app: Path
    ) -> None:
        assert autostart.is_enabled() is False
        autostart.enable()
        value = registry[(autostart.RUN_KEY, "EyeTracker")]
        assert value == subprocess.list2cmdline([str(frozen_app), "--background"])
        assert value.startswith(f'"{frozen_app}"')  # path contains spaces
        assert autostart.is_enabled() is True

    def test_enable_without_background(
        self, registry: dict[tuple[str, str], Any], frozen_app: Path
    ) -> None:
        autostart.enable(background=False)
        assert registry[(autostart.RUN_KEY, "EyeTracker")] == f'"{frozen_app}"'

    @pytest.mark.parametrize(
        ("flag", "enabled"),
        [(b"\x03\x00\x00\x00", False), (b"\x02\x00\x00\x00", True), (b"\x06\x00", True)],
    )
    def test_task_manager_disabled_flag(
        self, registry: dict[tuple[str, str], Any], flag: bytes, enabled: bool
    ) -> None:
        registry[(autostart.RUN_KEY, "EyeTracker")] = "x.exe"
        registry[(autostart.STARTUP_APPROVED_KEY, "EyeTracker")] = flag
        assert autostart.is_enabled() is enabled

    def test_enable_clears_task_manager_disabled_flag(
        self, registry: dict[tuple[str, str], Any]
    ) -> None:
        registry[(autostart.RUN_KEY, "EyeTracker")] = "old.exe"
        registry[(autostart.STARTUP_APPROVED_KEY, "EyeTracker")] = b"\x03" + bytes(11)
        autostart.enable()
        assert (autostart.STARTUP_APPROVED_KEY, "EyeTracker") not in registry
        assert autostart.is_enabled() is True

    def test_blank_value_is_not_enabled(self, registry: dict[tuple[str, str], Any]) -> None:
        registry[(autostart.RUN_KEY, "EyeTracker")] = "   "
        assert autostart.is_enabled() is False

    def test_disable_removes_everything(self, registry: dict[tuple[str, str], Any]) -> None:
        autostart.enable()
        registry[(autostart.STARTUP_APPROVED_KEY, "EyeTracker")] = b"\x03"
        registry[(autostart.RUN_KEY, "OtherApp")] = "other.exe"
        autostart.disable()
        assert registry == {(autostart.RUN_KEY, "OtherApp"): "other.exe"}
        assert autostart.is_enabled() is False
        autostart.disable()  # idempotent

    def test_registry_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def denied(*_args: Any) -> Any:
            raise PermissionError("access denied")

        monkeypatch.setattr(autostart, "_reg_read", denied)
        monkeypatch.setattr(autostart, "_reg_write_str", denied)
        monkeypatch.setattr(autostart, "_reg_delete", denied)
        assert autostart.is_enabled() is False
        with pytest.raises(autostart.AutostartError, match="access denied"):
            autostart.enable()
        with pytest.raises(OSError, match="disable"):
            autostart.disable()

    def test_location(self) -> None:
        assert autostart.location() == (
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run\EyeTracker"
        )


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows guard")
def test_real_registry_helpers_refuse_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # restore the real helpers; they must refuse before touching anything
    with pytest.raises(autostart.AutostartError):
        autostart._reg_read(autostart.RUN_KEY, "EyeTracker")


# ---------------------------------------------------------------------- macOS
@pytest.mark.usefixtures("frozen_app")
class TestMacOS:
    @pytest.fixture(autouse=True)
    def _macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_system(monkeypatch, "macos")

    def test_enable_writes_launch_agent(self, frozen_app: Path) -> None:
        autostart.enable()
        path = autostart._mac_plist_path()
        agent = plistlib.loads(path.read_bytes())
        assert agent == {
            "Label": APP_ID,
            "ProgramArguments": [str(frozen_app), "--background"],
            "RunAtLoad": True,
            "ProcessType": "Interactive",
            "LimitLoadToSessionType": "Aqua",
        }
        assert autostart.is_enabled() is True
        if sys.platform != "win32":
            assert path.stat().st_mode & 0o777 == 0o644

    def test_enable_overwrites(self, frozen_app: Path) -> None:
        autostart.enable()
        autostart.enable(background=False)
        agent = plistlib.loads(autostart._mac_plist_path().read_bytes())
        assert agent["ProgramArguments"] == [str(frozen_app)]

    def test_disabled_or_corrupt_agents_are_not_enabled(self) -> None:
        path = autostart._mac_plist_path()
        path.parent.mkdir(parents=True)
        path.write_bytes(
            plistlib.dumps({"Label": APP_ID, "ProgramArguments": ["x"], "Disabled": True})
        )
        assert autostart.is_enabled() is False
        path.write_text("<plist><dict><key>Label</oops>")
        assert autostart.is_enabled() is False
        path.write_bytes(plistlib.dumps({"Label": APP_ID, "ProgramArguments": []}))
        assert autostart.is_enabled() is False

    def test_disable(self) -> None:
        autostart.enable()
        autostart.disable()
        assert not autostart._mac_plist_path().exists()
        assert autostart.is_enabled() is False
        autostart.disable()

    def test_location_and_default_path(self) -> None:
        assert autostart.location() == str(autostart._mac_plist_path())
        default = REAL_MAC_PLIST_PATH()
        assert default == Path.home() / "Library" / "LaunchAgents" / f"{APP_ID}.plist"


# ---------------------------------------------------------------------- Linux
class TestLinux:
    @pytest.fixture(autouse=True)
    def _linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_system(monkeypatch, "linux")

    def test_enable_writes_desktop_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(paths, "is_frozen", lambda: True)
        image = tmp_path / "My Apps" / "Eye Tracker.AppImage"
        image.parent.mkdir()
        image.write_text("")
        monkeypatch.setenv("APPIMAGE", str(image))

        autostart.enable()
        path = autostart._linux_desktop_path()
        text = path.read_text(encoding="utf-8")
        entry = autostart._parse_desktop_entry(text)
        assert text.startswith("[Desktop Entry]\n")
        assert entry["Type"] == "Application"
        assert entry["Name"] == APP_NAME
        assert entry["Icon"] == APP_SLUG
        assert entry["Terminal"] == "false"
        assert entry["Hidden"] == "false"
        assert entry["NoDisplay"] == "false"
        assert entry["X-GNOME-Autostart-enabled"] == "true"
        assert desktop_exec_split(entry["Exec"]) == [str(image), "--background"]
        assert autostart.is_enabled() is True
        if sys.platform != "win32":
            assert path.stat().st_mode & 0o777 == 0o644

    @pytest.mark.parametrize(
        "line", ["Hidden=true", "X-GNOME-Autostart-enabled=false", "Hidden = True"]
    )
    def test_entries_switched_off_by_the_desktop(self, line: str) -> None:
        path = autostart._linux_desktop_path()
        path.parent.mkdir(parents=True)
        path.write_text(f"[Desktop Entry]\nType=Application\nExec=eye-tracker\n{line}\n")
        assert autostart.is_enabled() is False

    def test_entry_without_exec_is_not_enabled(self) -> None:
        path = autostart._linux_desktop_path()
        path.parent.mkdir(parents=True)
        path.write_text("[Desktop Entry]\nType=Application\n")
        assert autostart.is_enabled() is False

    def test_disable(self, source_python: Path) -> None:
        autostart.enable()
        assert autostart.is_enabled() is True
        autostart.disable()
        assert not autostart._linux_desktop_path().exists()
        autostart.disable()

    def test_location(self) -> None:
        assert autostart.location() == str(autostart._linux_desktop_path())

    def test_default_path_honours_xdg_config_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
        assert REAL_LINUX_DESKTOP_PATH() == tmp_path / "cfg" / "autostart" / "eye-tracker.desktop"
        monkeypatch.setenv("XDG_CONFIG_HOME", "relative/dir")  # invalid per spec -> ignored
        assert REAL_LINUX_DESKTOP_PATH() == (
            Path.home() / ".config" / "autostart" / "eye-tracker.desktop"
        )
        monkeypatch.delenv("XDG_CONFIG_HOME")
        assert REAL_LINUX_DESKTOP_PATH().parent == Path.home() / ".config" / "autostart"


class TestDesktopExecQuoting:
    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            ("/usr/bin/python3", "/usr/bin/python3"),
            ("--background", "--background"),
            ("/opt/My App/run", '"/opt/My App/run"'),
            ('say "hi"', r'"say \"hi\""'),
            ("/opt/a$b", r'"/opt/a\$b"'),
            ("back`tick", r'"back\`tick"'),
            (r"C:\x", r'"C:\\x"'),
            ("100%", "100%%"),
            ("", '""'),
        ],
    )
    def test_argument_quoting(self, arg: str, expected: str) -> None:
        assert autostart._desktop_exec_arg(arg) == expected

    def test_string_escaping_is_applied_on_top(self) -> None:
        # Spec example: a literal "$" inside quotes is written as \\$ in the file.
        assert autostart._desktop_exec(["/opt/a$b"]) == '"/opt/a\\\\$b"'

    @pytest.mark.parametrize(
        "command",
        [
            ["/usr/bin/python3", "-m", "eye_tracker", "--background"],
            ["/home/me/My Apps/Eye Tracker.AppImage", "--background"],
            ['/opt/we$ird `name`/app "q" 50%', "a\\b", "x;y|z"],
        ],
    )
    def test_round_trip(self, command: list[str]) -> None:
        assert desktop_exec_split(autostart._desktop_exec(command)) == command

    def test_parse_desktop_entry_only_reads_main_group(self) -> None:
        text = (
            "# comment\n[Desktop Entry]\nName = Eye Tracker\nExec=a\n"
            "[Desktop Action x]\nExec=b\nHidden=true\n"
        )
        entry = autostart._parse_desktop_entry(text)
        assert entry == {"Name": "Eye Tracker", "Exec": "a"}


# ---------------------------------------------------------------- unsupported
class TestUnsupported:
    @pytest.fixture(autouse=True)
    def _other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_system(monkeypatch, "unsupported")

    def test_everything_degrades(self) -> None:
        assert autostart.is_supported() is False
        assert autostart.is_enabled() is False
        assert autostart.location() is None
        with pytest.raises(autostart.AutostartError, match="not supported"):
            autostart.enable()
        autostart.disable()  # no-op


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [("win32", "windows"), ("darwin", "macos"), ("linux", "linux"), ("freebsd14", "unsupported")],
)
def test_system_detection(platform_name: str, expected: str) -> None:
    assert autostart._system(platform_name) == expected


def test_system_defaults_to_running_os() -> None:
    assert autostart._system() == autostart._system(sys.platform)
