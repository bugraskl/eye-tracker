"""Tests for ``eye_tracker.platform.autostart`` on all three OS code paths.

The OS is selected by patching ``autostart._system``; the registry helpers are
replaced by an in-memory dict and the LaunchAgent / .desktop locations point
into ``tmp_path``. An autouse fixture makes any accidental use of the real
registry helpers fail, so the real ``HKCU\\...\\Run`` key is never touched.
"""

from __future__ import annotations

import logging
import plistlib
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from eye_tracker import APP_ID, APP_NAME, APP_SLUG, cli, paths
from eye_tracker.platform import autostart

REAL_LINUX_DESKTOP_PATH = autostart._linux_desktop_path
REAL_MAC_PLIST_PATH = autostart._mac_plist_path
Status = autostart.Status


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
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
    paths.set_base_override(None)  # the default profile unless a test says otherwise
    yield
    paths.set_base_override(None)


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
        self, registry: dict[tuple[str, str], Any], frozen_app: Path, flag: bytes, enabled: bool
    ) -> None:
        registry[(autostart.RUN_KEY, "EyeTracker")] = f'"{frozen_app}" --background'
        registry[(autostart.STARTUP_APPROVED_KEY, "EyeTracker")] = flag
        assert autostart.is_enabled() is enabled
        expected = autostart.Status.ENABLED if enabled else autostart.Status.DISABLED
        assert autostart.status() is expected

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
        assert autostart.status() is Status.DISABLED
        assert autostart.registered_command() is None
        assert autostart.refresh() is False
        assert autostart.location() is None
        with pytest.raises(autostart.AutostartError, match="not supported"):
            autostart.enable()
        autostart.disable()  # no-op


# ------------------------------------------------------------------- profiles
def run_value(registry: dict[tuple[str, str], Any]) -> list[str]:
    return autostart._win_split(registry[(autostart.RUN_KEY, "EyeTracker")])


@pytest.fixture
def portable(tmp_path: Path) -> Path:
    """A ``--config-dir`` profile directory."""
    directory = tmp_path / "Portable Data"
    directory.mkdir()
    return directory.resolve()


@pytest.mark.usefixtures("frozen_app")
class TestProfiles:
    """--config-dir survives into the login entry (portable mode)."""

    def test_launch_command_carries_the_profile(
        self, monkeypatch: pytest.MonkeyPatch, frozen_app: Path, portable: Path
    ) -> None:
        use_system(monkeypatch, "windows")
        assert autostart.launch_command(config_dir=portable) == [
            str(frozen_app),
            "--config-dir",
            str(portable),
            "--background",
        ]
        # Relative or "~" paths are resolved the way paths.set_base_override does.
        monkeypatch.chdir(portable.parent)
        assert autostart.launch_command(False, Path(portable.name))[1:] == [
            "--config-dir",
            str(portable),
        ]

    def test_default_is_the_running_profile(
        self, monkeypatch: pytest.MonkeyPatch, frozen_app: Path, portable: Path
    ) -> None:
        use_system(monkeypatch, "windows")
        assert autostart.launch_command() == [str(frozen_app), "--background"]
        paths.set_base_override(portable)
        assert autostart.launch_command() == autostart.launch_command(config_dir=portable)

    def test_the_cli_parses_the_login_command(
        self, monkeypatch: pytest.MonkeyPatch, portable: Path
    ) -> None:
        use_system(monkeypatch, "windows")
        args = cli.build_parser().parse_args(autostart.launch_command(config_dir=portable)[1:])
        assert args.config_dir == str(portable)
        assert args.background is True
        assert args.command is None  # starts the tray app

    def test_windows_entry_is_per_profile(
        self,
        monkeypatch: pytest.MonkeyPatch,
        registry: dict[tuple[str, str], Any],
        frozen_app: Path,
        portable: Path,
    ) -> None:
        use_system(monkeypatch, "windows")
        autostart.enable(config_dir=portable)
        assert run_value(registry) == autostart.launch_command(config_dir=portable)
        assert autostart.status(portable) is Status.ENABLED
        assert autostart.is_enabled(portable) is True
        # The installed (default-profile) app does not claim the portable entry...
        assert autostart.status() is Status.OTHER_PROFILE
        assert autostart.is_enabled() is False
        # ...and does not remove it either.
        autostart.disable()
        assert autostart.status(portable) is Status.ENABLED
        # Enabling it there takes the single login entry over (one tracker per camera).
        autostart.enable()
        assert run_value(registry) == [str(frozen_app), "--background"]
        assert autostart.status(portable) is Status.OTHER_PROFILE
        autostart.disable()
        assert registry == {}

    def test_running_portable_instance_uses_its_profile(
        self, monkeypatch: pytest.MonkeyPatch, registry: dict[tuple[str, str], Any], portable: Path
    ) -> None:
        """The tray/Settings/wizard calls need no argument in a --config-dir process."""
        use_system(monkeypatch, "windows")
        paths.set_base_override(portable)
        autostart.enable(background=True)
        assert run_value(registry)[1:] == ["--config-dir", str(portable), "--background"]
        assert autostart.is_enabled() is True
        autostart.disable()
        assert autostart.status() is Status.DISABLED

    def test_windows_profile_comparison_ignores_case(
        self,
        monkeypatch: pytest.MonkeyPatch,
        registry: dict[tuple[str, str], Any],
        frozen_app: Path,
        portable: Path,
    ) -> None:
        use_system(monkeypatch, "windows")
        command = [str(frozen_app), "--config-dir", str(portable).upper(), "--background"]
        registry[(autostart.RUN_KEY, "EyeTracker")] = subprocess.list2cmdline(command)
        assert autostart.status(portable) is Status.ENABLED

    def test_equals_form_is_understood(
        self,
        monkeypatch: pytest.MonkeyPatch,
        registry: dict[tuple[str, str], Any],
        frozen_app: Path,
        portable: Path,
    ) -> None:
        use_system(monkeypatch, "windows")
        command = [str(frozen_app), f"--config-dir={portable}"]
        registry[(autostart.RUN_KEY, "EyeTracker")] = subprocess.list2cmdline(command)
        assert autostart.status(portable) is Status.ENABLED
        assert autostart.status() is Status.OTHER_PROFILE

    def test_macos_and_linux_entries_carry_the_profile(
        self, monkeypatch: pytest.MonkeyPatch, frozen_app: Path, portable: Path
    ) -> None:
        use_system(monkeypatch, "macos")
        autostart.enable(config_dir=portable)
        agent = plistlib.loads(autostart._mac_plist_path().read_bytes())
        assert agent["ProgramArguments"] == autostart.launch_command(config_dir=portable)
        assert autostart.status(portable) is Status.ENABLED
        assert autostart.status() is Status.OTHER_PROFILE

        use_system(monkeypatch, "linux")
        autostart.enable(config_dir=portable)
        assert autostart.registered_command() == autostart.launch_command(config_dir=portable)
        assert autostart.status(portable) is Status.ENABLED
        assert autostart.status() is Status.OTHER_PROFILE


# --------------------------------------------------------------- stale entries
MAC_APP = "/Applications/Eye Tracker.app/Contents/MacOS/Eye Tracker"
TRANSLOCATED = (
    "/private/var/folders/ab/xyz/T/AppTranslocation/0A1B-2C3D/d/"
    "Eye Tracker.app/Contents/MacOS/Eye Tracker"
)
ON_DMG = "/Volumes/Eye Tracker/Eye Tracker.app/Contents/MacOS/Eye Tracker"


def run_frozen_mac_app(monkeypatch: pytest.MonkeyPatch, executable: str) -> None:
    """Pretend to be the bundled macOS app at ``executable`` (a POSIX path on any host)."""
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(autostart, "_frozen_gui_executable", lambda: PurePosixPath(executable))


def write_agent(arguments: list[str]) -> None:
    path = autostart._mac_plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps({"Label": APP_ID, "ProgramArguments": arguments}))


def write_desktop_entry(command: list[str], *, hidden: bool = False) -> None:
    path = autostart._linux_desktop_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = autostart._desktop_entry(command)
    if hidden:  # what GNOME Tweaks / KDE do to switch an entry off
        text = text.replace("Hidden=false", "Hidden=true")
    path.write_text(text)


class TestStaleEntries:
    def test_windows_entry_for_a_missing_program(
        self, monkeypatch: pytest.MonkeyPatch, registry: dict[tuple[str, str], Any], tmp_path: Path
    ) -> None:
        use_system(monkeypatch, "windows")
        gone = tmp_path / "old portable" / "EyeTracker.exe"
        registry[(autostart.RUN_KEY, "EyeTracker")] = f'"{gone}" --background'
        assert autostart.status() is Status.STALE
        assert autostart.is_enabled() is False
        assert autostart.registered_command() == [str(gone), "--background"]

    def test_program_found_on_path(
        self, monkeypatch: pytest.MonkeyPatch, registry: dict[tuple[str, str], Any]
    ) -> None:
        use_system(monkeypatch, "windows")
        monkeypatch.setattr(autostart.shutil, "which", lambda name: f"/usr/bin/{name}")
        registry[(autostart.RUN_KEY, "EyeTracker")] = "eye-tracker-gui --background"
        assert autostart.status() is Status.ENABLED
        monkeypatch.setattr(autostart.shutil, "which", lambda name: None)
        assert autostart.status() is Status.STALE

    def test_linux_entry_for_a_replaced_appimage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        use_system(monkeypatch, "linux")
        write_desktop_entry([str(tmp_path / "EyeTracker-0.1.0-linux-x86_64.AppImage")])
        assert autostart.status() is Status.STALE

    def test_macos_entry_in_a_temporary_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_system(monkeypatch, "macos")
        write_agent([TRANSLOCATED, "--background"])
        assert autostart.status() is Status.STALE
        monkeypatch.setattr(autostart, "_is_read_only_volume", lambda path: True)
        write_agent([ON_DMG, "--background"])
        assert autostart.status() is Status.STALE

    @pytest.mark.parametrize("executable", [TRANSLOCATED, ON_DMG])
    def test_macos_refuses_to_register_a_temporary_copy(
        self, monkeypatch: pytest.MonkeyPatch, executable: str
    ) -> None:
        use_system(monkeypatch, "macos")
        run_frozen_mac_app(monkeypatch, executable)
        monkeypatch.setattr(autostart, "_is_read_only_volume", lambda path: True)
        with pytest.raises(autostart.AutostartError, match="Applications folder"):
            autostart.enable()
        assert not autostart._mac_plist_path().exists()

    def test_macos_app_on_a_writable_external_volume_is_fine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        use_system(monkeypatch, "macos")
        run_frozen_mac_app(monkeypatch, ON_DMG)
        monkeypatch.setattr(autostart, "_is_read_only_volume", lambda path: False)
        autostart.enable()
        assert autostart.registered_command() == [ON_DMG, "--background"]

    def test_transient_locations_only_exist_on_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_system(monkeypatch, "linux")
        assert autostart._is_transient_location(TRANSLOCATED) is False

    def test_read_only_volume_check(self, tmp_path: Path) -> None:
        assert autostart._is_read_only_volume(str(tmp_path)) is False  # writable (or no statvfs)
        assert autostart._is_read_only_volume(str(tmp_path / "missing")) is False


# --------------------------------------------------------------------- refresh
class TestRefresh:
    """The app calls refresh() at startup to follow a moved or upgraded copy."""

    @pytest.fixture
    def appimage(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Running from EyeTracker-0.2.0 as a frozen AppImage build."""
        use_system(monkeypatch, "linux")
        monkeypatch.setattr(paths, "is_frozen", lambda: True)
        image = tmp_path / "Apps" / "EyeTracker-0.2.0-linux-x86_64.AppImage"
        image.parent.mkdir()
        image.write_text("")
        monkeypatch.setenv("APPIMAGE", str(image))
        return image

    def test_replaced_appimage_is_followed(self, appimage: Path) -> None:
        old = appimage.with_name("EyeTracker-0.1.0-linux-x86_64.AppImage")  # deleted
        write_desktop_entry([str(old), "--background"])
        assert autostart.refresh() is True
        assert autostart.registered_command() == [str(appimage), "--background"]
        assert autostart.status() is Status.ENABLED

    def test_packaged_build_replaces_an_older_copy_that_still_exists(self, appimage: Path) -> None:
        old = appimage.with_name("EyeTracker-0.1.0-linux-x86_64.AppImage")
        old.write_text("")
        write_desktop_entry([str(old), "--background"])
        assert autostart.refresh() is True
        assert autostart.registered_command() == [str(appimage), "--background"]

    def test_background_choice_and_profile_are_kept(self, appimage: Path, portable: Path) -> None:
        old = appimage.with_name("gone.AppImage")
        write_desktop_entry([str(old), "--config-dir", str(portable)])
        assert autostart.refresh() is False  # another profile's entry: not ours to move
        assert autostart.refresh(portable) is True
        assert autostart.registered_command() == [str(appimage), "--config-dir", str(portable)]

    def test_up_to_date_entry_is_not_rewritten(self, appimage: Path) -> None:
        autostart.enable()
        before = autostart._linux_desktop_path().stat().st_mtime_ns
        assert autostart.refresh() is False
        assert autostart._linux_desktop_path().stat().st_mtime_ns == before

    def test_switched_off_entry_is_left_alone(self, appimage: Path) -> None:
        old = str(appimage.with_name("gone.AppImage"))
        write_desktop_entry([old], hidden=True)
        assert autostart.status() is Status.DISABLED
        assert autostart.refresh() is False
        assert autostart.registered_command() == [old]

    def test_no_entry_is_not_created(self, appimage: Path) -> None:
        assert autostart.refresh() is False
        assert not autostart._linux_desktop_path().exists()

    def test_source_run_does_not_take_over_a_working_entry(
        self, monkeypatch: pytest.MonkeyPatch, source_python: Path, tmp_path: Path
    ) -> None:
        """A developer's checkout must not steal the installed app's entry."""
        use_system(monkeypatch, "linux")
        installed = tmp_path / "opt" / "EyeTracker.AppImage"
        installed.parent.mkdir()
        installed.write_text("")
        write_desktop_entry([str(installed), "--background"])
        assert autostart.refresh() is False
        assert autostart.registered_command() == [str(installed), "--background"]

    def test_source_run_repairs_an_entry_of_a_deleted_environment(
        self, monkeypatch: pytest.MonkeyPatch, source_python: Path, tmp_path: Path
    ) -> None:
        use_system(monkeypatch, "linux")
        gone = str(tmp_path / "old-venv" / "bin" / "python")
        write_desktop_entry([gone, "-m", "eye_tracker", "--background"])
        assert autostart.refresh() is True
        assert autostart.registered_command() == autostart.launch_command()

    def test_windows_installer_entry_matches_despite_case(
        self,
        monkeypatch: pytest.MonkeyPatch,
        registry: dict[tuple[str, str], Any],
        frozen_app: Path,
    ) -> None:
        use_system(monkeypatch, "windows")
        installer_value = f'"{str(frozen_app).upper()}" --background'
        registry[(autostart.RUN_KEY, "EyeTracker")] = installer_value
        registry[(autostart.STARTUP_APPROVED_KEY, "EyeTracker")] = b"\x02" + bytes(11)
        assert autostart.refresh() is False
        assert registry[(autostart.RUN_KEY, "EyeTracker")] == installer_value
        # The Task Manager state is untouched as well.
        assert (autostart.STARTUP_APPROVED_KEY, "EyeTracker") in registry

    def test_task_manager_disabled_entry_is_not_revived(
        self,
        monkeypatch: pytest.MonkeyPatch,
        registry: dict[tuple[str, str], Any],
        frozen_app: Path,
        tmp_path: Path,
    ) -> None:
        use_system(monkeypatch, "windows")
        registry[(autostart.RUN_KEY, "EyeTracker")] = f'"{tmp_path / "gone.exe"}"'
        registry[(autostart.STARTUP_APPROVED_KEY, "EyeTracker")] = b"\x03" + bytes(11)
        assert autostart.refresh() is False
        assert registry[(autostart.STARTUP_APPROVED_KEY, "EyeTracker")] == b"\x03" + bytes(11)

    def test_macos_entry_made_from_the_disk_image_is_moved_to_applications(
        self, monkeypatch: pytest.MonkeyPatch, frozen_app: Path
    ) -> None:
        use_system(monkeypatch, "macos")
        write_agent([TRANSLOCATED, "--background"])
        assert autostart.refresh() is True
        assert autostart.registered_command() == [str(frozen_app), "--background"]

    def test_never_points_the_entry_at_a_temporary_copy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        use_system(monkeypatch, "macos")
        run_frozen_mac_app(monkeypatch, TRANSLOCATED)
        gone = str(tmp_path / "Eye Tracker.app" / "Contents" / "MacOS" / "Eye Tracker")
        write_agent([gone, "--background"])
        assert autostart.refresh() is False
        assert autostart.registered_command() == [gone, "--background"]

    def test_failures_are_logged_not_raised(
        self,
        appimage: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        write_desktop_entry([str(appimage.with_name("gone.AppImage"))])

        def read_only(path: Path, text: str) -> None:
            raise PermissionError("read-only home")

        monkeypatch.setattr(autostart, "_write_text", read_only)
        with caplog.at_level(logging.WARNING, logger="eye_tracker.platform.autostart"):
            assert autostart.refresh() is False
        assert "Could not update" in caplog.text


# ----------------------------------------------------------- command parsing
WINDOWS_COMMANDS = [
    [r"C:\Program Files\Eye Tracker\EyeTracker.exe", "--background"],
    [r"C:\Python\pythonw.exe", "-m", "eye_tracker", "--config-dir", r"D:\My Data\ET"],
    [r"C:\x\app.exe", "--config-dir", "D:\\trailing slash\\"],
    [r"C:\x\app.exe", 'say "hi"', "back\\\\slashes", "", "tab\there", 'a\\"b'],
    [r"C:\no-spaces\app.exe"],
]


class TestCommandParsing:
    @pytest.mark.parametrize("command", WINDOWS_COMMANDS)
    def test_windows_round_trip(self, command: list[str]) -> None:
        assert autostart._win_split(subprocess.list2cmdline(command)) == command

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ('"C:\\a b\\x.exe" --background', ["C:\\a b\\x.exe", "--background"]),
            ("  x.exe   a\tb  ", ["x.exe", "a", "b"]),
            ('x.exe "a""b"', ["x.exe", 'a"b']),
            ('x.exe a\\\\"b c"', ["x.exe", "a\\b c"]),
            ('"C:\\unterminated', ["C:\\unterminated"]),
            ("", []),
        ],
    )
    def test_windows_rules(self, line: str, expected: list[str]) -> None:
        assert autostart._win_split(line) == expected

    @pytest.mark.skipif(sys.platform != "win32", reason="compares with the Windows parser")
    @pytest.mark.parametrize("command", WINDOWS_COMMANDS)
    def test_windows_split_agrees_with_command_line_to_argv(self, command: list[str]) -> None:
        import ctypes
        from ctypes import wintypes

        shell32 = ctypes.WinDLL("shell32")
        kernel32 = ctypes.WinDLL("kernel32")
        shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
        shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
        kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
        kernel32.LocalFree.restype = wintypes.HLOCAL
        line = subprocess.list2cmdline(command)
        count = ctypes.c_int(0)
        argv = shell32.CommandLineToArgvW(line, ctypes.byref(count))
        try:
            reference = [argv[i] for i in range(count.value)]
        finally:
            kernel32.LocalFree(ctypes.cast(argv, wintypes.HLOCAL))
        assert autostart._win_split(line) == reference

    @pytest.mark.parametrize(
        "command",
        [
            ["/usr/bin/python3", "-m", "eye_tracker", "--background"],
            ["/home/me/My Apps/Eye Tracker.AppImage", "--config-dir", "/home/me/ET data"],
            ['/opt/we$ird `name`/app "q" 50%', "a\\b", "x;y|z", "tab\there"],
        ],
    )
    def test_desktop_exec_round_trip(self, command: list[str]) -> None:
        value = autostart._desktop_exec(command)
        assert autostart._desktop_exec_split(value) == command
        assert autostart._desktop_exec_split(value) == desktop_exec_split(value)

    def test_desktop_exec_split_is_lenient_with_foreign_entries(self) -> None:
        assert autostart._desktop_exec_split('sh -c "echo \\"hi\\""') == ["sh", "-c", 'echo "hi"']
        assert autostart._desktop_exec_split("app %U") == ["app", "%U"]

    def test_profile_helpers(self) -> None:
        command = ["app", "--config-dir", "/a", "--background", "--config-dir=/b"]
        assert autostart._command_profile(command) == "/a"
        assert autostart._without_profile(command) == ["--background"]
        assert autostart._command_profile(["app", "--config-dir"]) is None  # no value
        assert autostart._command_profile(["--config-dir", "/program/is/first"]) is None


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [("win32", "windows"), ("darwin", "macos"), ("linux", "linux"), ("freebsd14", "unsupported")],
)
def test_system_detection(platform_name: str, expected: str) -> None:
    assert autostart._system(platform_name) == expected


def test_system_defaults_to_running_os() -> None:
    assert autostart._system() == autostart._system(sys.platform)
