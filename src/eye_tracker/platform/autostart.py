"""Start at login ("autostart") on Windows, macOS and Linux.

Each OS uses its native per-user mechanism; none needs administrator rights:

* **Windows**: a ``HKCU\\...\\CurrentVersion\\Run`` value named ``EyeTracker``
  (the same value the installer's "Start at login" task writes).
* **macOS**: a LaunchAgent plist in ``~/Library/LaunchAgents``. It is only
  written, never loaded with ``launchctl``: launchd picks it up at next login,
  which is exactly when it is needed.
* **Linux**: an XDG autostart entry, ``$XDG_CONFIG_HOME/autostart/eye-tracker.desktop``.

Filesystem locations and registry access go through small module-level helpers
(``_mac_plist_path``, ``_linux_desktop_path``, ``_reg_read``/``_reg_write_str``/
``_reg_delete``) so tests can redirect them.
"""

from __future__ import annotations

import logging
import os
import plistlib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import APP_ID, APP_NAME, APP_SLUG, paths
from ..config import atomic_write_text

log = logging.getLogger(__name__)

__all__ = [
    "BACKGROUND_FLAG",
    "AutostartError",
    "disable",
    "enable",
    "is_enabled",
    "is_supported",
    "launch_command",
    "location",
]

BACKGROUND_FLAG = "--background"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
#: Where Task Manager's "Startup apps" page records entries the user disabled.
STARTUP_APPROVED_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"
RUN_VALUE_NAME = "EyeTracker"
LAUNCH_AGENT_LABEL = APP_ID
DESKTOP_FILE_NAME = f"{APP_SLUG}.desktop"

#: Names the windowed executable may have next to a console ``*-cli`` executable.
_GUI_EXECUTABLE_NAMES = ("EyeTracker.exe", "EyeTracker", "Eye Tracker", APP_SLUG)
#: Characters that force quoting inside a Desktop Entry ``Exec`` value.
_DESKTOP_RESERVED = frozenset(" \t\n\"'\\><~|&;$*?#()`")
#: Characters escaped with a backslash inside a quoted ``Exec`` argument.
_DESKTOP_QUOTED_ESCAPES = frozenset('"`$\\')


class AutostartError(OSError):
    """Enabling or disabling start at login failed (message is user-presentable)."""


# ------------------------------------------------------------------ public API
def is_supported() -> bool:
    """Whether this OS has an autostart implementation."""
    return _system() in {"windows", "macos", "linux"}


def launch_command(background: bool = True) -> list[str]:
    """Command line that starts this installation of the app.

    * frozen build inside an AppImage: the ``.AppImage`` file (the mounted
      bundle path changes on every run);
    * other frozen builds: the windowed executable (inside the ``.app`` bundle
      on macOS);
    * source/pip installs: ``python -m eye_tracker``, using ``pythonw.exe`` on
      Windows so no console window opens at login.

    ``background`` appends ``--background`` (no first-run wizard or startup
    notifications), which is what a login launch wants.
    """
    command = _base_command()
    if background:
        command.append(BACKGROUND_FLAG)
    return command


def is_enabled() -> bool:
    """Whether the app is registered to start at login. Never raises."""
    system = _system()
    try:
        if system == "windows":
            return _win_is_enabled()
        if system == "macos":
            return _mac_is_enabled()
        if system == "linux":
            return _linux_is_enabled()
    except Exception:
        log.warning("Could not read the start-at-login state", exc_info=True)
    return False


def enable(background: bool = True) -> None:
    """Register the app to start at login (replacing any previous entry).

    Raises :class:`AutostartError` on failure or on an unsupported OS.
    """
    system = _system()
    try:
        if system == "windows":
            _win_enable(background)
        elif system == "macos":
            _mac_enable(background)
        elif system == "linux":
            _linux_enable(background)
        else:
            raise AutostartError(f"Start at login is not supported on {sys.platform}")
    except AutostartError:
        raise
    except Exception as exc:
        raise AutostartError(f"Could not enable start at login: {exc}") from exc
    log.info("Start at login enabled (%s)", location())


def disable() -> None:
    """Remove the login entry. A missing entry is not an error.

    Raises :class:`AutostartError` when an existing entry cannot be removed.
    """
    system = _system()
    try:
        if system == "windows":
            _win_disable()
        elif system == "macos":
            _unlink(_mac_plist_path())
        elif system == "linux":
            _unlink(_linux_desktop_path())
        else:
            return
    except Exception as exc:
        raise AutostartError(f"Could not disable start at login: {exc}") from exc
    log.info("Start at login disabled")


def location() -> str | None:
    """Human-readable location of the login entry (for diagnostics)."""
    system = _system()
    if system == "windows":
        return f"HKCU\\{RUN_KEY}\\{RUN_VALUE_NAME}"
    if system == "macos":
        return str(_mac_plist_path())
    if system == "linux":
        return str(_linux_desktop_path())
    return None


# -------------------------------------------------------------------- command
def _system(platform: str | None = None) -> str:
    """``"windows"``, ``"macos"``, ``"linux"`` or ``"unsupported"`` (tests patch this)."""
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return "windows"
    if platform == "darwin":
        return "macos"
    if platform.startswith("linux"):
        return "linux"
    return "unsupported"


def _base_command() -> list[str]:
    if paths.is_frozen():
        # Only trusted when frozen: APPIMAGE is inherited by every child of an
        # AppImage (a terminal emulator AppImage, for instance).
        appimage = os.environ.get("APPIMAGE", "")
        if _system() == "linux" and appimage and Path(appimage).is_file():
            return [appimage]
        return [str(_frozen_gui_executable())]
    return [_source_interpreter(), "-m", "eye_tracker"]


def _frozen_gui_executable() -> Path:
    """The windowed executable, even when called from the console ``*-cli`` one.

    Registering the console build would open a terminal window at every login.
    """
    exe = Path(sys.executable)
    if exe.stem.lower().endswith("-cli"):
        for name in _GUI_EXECUTABLE_NAMES:
            candidate = exe.with_name(name)
            if candidate != exe and candidate.is_file():
                return candidate
    return exe


def _source_interpreter() -> str:
    exe = Path(sys.executable)
    if _system() == "windows" and exe.name.lower() == "python.exe":
        pythonw = exe.with_name("pythonw.exe")
        if pythonw.is_file():
            return str(pythonw)
    return str(exe)


# -------------------------------------------------------------------- Windows
def _win_command_line(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def _win_is_enabled() -> bool:
    value = _reg_read(RUN_KEY, RUN_VALUE_NAME)
    if not isinstance(value, str) or not value.strip():
        return False
    return not _win_disabled_in_task_manager()


def _win_disabled_in_task_manager() -> bool:
    """Task Manager keeps the Run value but flags it; the first byte is odd when disabled."""
    data = _reg_read(STARTUP_APPROVED_KEY, RUN_VALUE_NAME)
    return isinstance(data, (bytes, bytearray)) and len(data) > 0 and data[0] & 1 == 1


def _win_enable(background: bool) -> None:
    _reg_write_str(RUN_KEY, RUN_VALUE_NAME, _win_command_line(launch_command(background)))
    # Clearing Task Manager's "disabled" flag makes the new entry effective;
    # otherwise the toggle in our UI would silently do nothing.
    _reg_delete(STARTUP_APPROVED_KEY, RUN_VALUE_NAME)


def _win_disable() -> None:
    _reg_delete(RUN_KEY, RUN_VALUE_NAME)
    _reg_delete(STARTUP_APPROVED_KEY, RUN_VALUE_NAME)


def _reg_read(key_path: str, name: str) -> Any | None:
    """Value ``name`` under ``HKCU\\key_path``, ``None`` when missing."""
    if sys.platform != "win32":
        raise AutostartError("the Windows registry is not available on this OS")
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ) as key:
            value, _kind = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return None
    return value


def _reg_write_str(key_path: str, name: str, value: str) -> None:
    """Write a ``REG_SZ`` value under ``HKCU\\key_path`` (creating the key)."""
    if sys.platform != "win32":
        raise AutostartError("the Windows registry is not available on this OS")
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)


def _reg_delete(key_path: str, name: str) -> None:
    """Delete value ``name`` under ``HKCU\\key_path``; missing is fine."""
    if sys.platform != "win32":
        raise AutostartError("the Windows registry is not available on this OS")
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, name)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------- macOS
def _mac_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"


def _mac_plist(command: Sequence[str]) -> bytes:
    agent = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": list(command),
        "RunAtLoad": True,
        # A user-facing app: no background QoS throttling of the camera loop.
        "ProcessType": "Interactive",
        # Only in the graphical login session (not SSH or the login window).
        "LimitLoadToSessionType": "Aqua",
    }
    return plistlib.dumps(agent, fmt=plistlib.FMT_XML, sort_keys=True)


def _mac_enable(background: bool) -> None:
    path = _mac_plist_path()
    _write_text(path, _mac_plist(launch_command(background)).decode("utf-8"))


def _mac_is_enabled() -> bool:
    path = _mac_plist_path()
    try:
        agent = plistlib.loads(path.read_bytes())
    except FileNotFoundError:
        return False
    except Exception:
        log.debug("Unreadable LaunchAgent %s", path, exc_info=True)
        return False  # launchd would ignore it as well
    if not isinstance(agent, dict) or agent.get("Disabled") is True:
        return False
    arguments = agent.get("ProgramArguments")
    return isinstance(arguments, list) and bool(arguments)


# ---------------------------------------------------------------------- Linux
def _linux_desktop_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", "")
    # The XDG spec says relative values are invalid and must be ignored.
    root = Path(base) if base and os.path.isabs(base) else Path.home() / ".config"
    return root / "autostart" / DESKTOP_FILE_NAME


def _desktop_exec_arg(arg: str) -> str:
    """Quote one argument following the Desktop Entry ``Exec`` rules.

    Only double quotes are recognised (``shlex.quote``'s single quotes are
    not), and ``%`` must be doubled because it introduces field codes.
    """
    if arg and not any(ch in _DESKTOP_RESERVED for ch in arg):
        quoted = arg
    else:
        escaped = "".join(f"\\{ch}" if ch in _DESKTOP_QUOTED_ESCAPES else ch for ch in arg)
        quoted = f'"{escaped}"'
    return quoted.replace("%", "%%")


def _desktop_string_escape(value: str) -> str:
    """Escaping of the key file's ``string`` type, applied on top of ``Exec`` quoting."""
    return (
        value.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    )


def _desktop_exec(command: Sequence[str]) -> str:
    return _desktop_string_escape(" ".join(_desktop_exec_arg(arg) for arg in command))


def _desktop_entry(command: Sequence[str]) -> str:
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.5",
        f"Name={APP_NAME}",
        "Comment=Moves the mouse cursor to the monitor you are looking at",
        f"Exec={_desktop_exec(command)}",
        f"Icon={APP_SLUG}",
        "Terminal=false",
        "Hidden=false",
        "NoDisplay=false",
        "X-GNOME-Autostart-enabled=true",
        # Give the panel a moment so the tray icon has somewhere to go.
        "X-GNOME-Autostart-Delay=3",
    ]
    return "\n".join(lines) + "\n"


def _parse_desktop_entry(text: str) -> dict[str, str]:
    """Keys of the ``[Desktop Entry]`` group (values unescaped only as far as needed here)."""
    entry: dict[str, str] = {}
    group = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            group = line[1:-1]
            continue
        if group == "Desktop Entry" and "=" in line:
            key, _, value = line.partition("=")
            entry.setdefault(key.strip(), value.strip())
    return entry


def _linux_enable(background: bool) -> None:
    _write_text(_linux_desktop_path(), _desktop_entry(launch_command(background)))


def _linux_is_enabled() -> bool:
    path = _linux_desktop_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    entry = _parse_desktop_entry(text)
    # Desktop environments "turn off" an autostart entry by editing these keys
    # (GNOME Tweaks, KDE System Settings) rather than deleting the file.
    if entry.get("Hidden", "false").lower() == "true":
        return False
    if entry.get("X-GNOME-Autostart-enabled", "true").lower() == "false":
        return False
    return bool(entry.get("Exec"))


# ------------------------------------------------------------------- helpers
def _write_text(path: Path, text: str) -> None:
    atomic_write_text(path, text)
    try:
        # mkstemp creates 0600 files; login entries are conventionally 0644.
        path.chmod(0o644)
    except OSError:
        log.debug("Could not chmod %s", path, exc_info=True)


def _unlink(path: Path) -> None:
    path.unlink(missing_ok=True)
