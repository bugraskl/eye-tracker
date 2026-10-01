"""Start at login ("autostart") on Windows, macOS and Linux.

Each OS uses its native per-user mechanism; none needs administrator rights:

* **Windows**: a ``HKCU\\...\\CurrentVersion\\Run`` value named ``EyeTracker``
  (the same value the installer's "Start at login" task writes).
* **macOS**: a LaunchAgent plist in ``~/Library/LaunchAgents``. It is only
  written, never loaded with ``launchctl``: launchd picks it up at next login,
  which is exactly when it is needed. A switch-off that launchd records in its
  own database (``launchctl disable``, possibly the Login Items switch in
  System Settings) leaves the plist untouched; it is read back with
  ``launchctl print-disabled`` and cleared by :func:`enable` with
  ``launchctl enable``, which does not load the agent either.
* **Linux**: an XDG autostart entry, ``$XDG_CONFIG_HOME/autostart/eye-tracker.desktop``.

Profiles
--------
Every function takes ``config_dir``: the ``--config-dir`` profile the entry is
for. ``None`` means the profile of the running process (its ``--config-dir``,
if it was started with one). A non-default profile's entry carries
``--config-dir <dir>`` in its command, so the login launch opens the same
settings and calibration and is recognised as the same instance.

There is still only one entry per user: the camera and the cursor can serve
one tracker, so turning start at login on in one profile replaces another
profile's entry, and :func:`status` reports such an entry as
:attr:`Status.OTHER_PROFILE` rather than as enabled.

Stale entries
-------------
An entry records the path of the program that wrote it. That path can go away:
a replaced AppImage, a moved portable folder, a deleted virtual environment, or
a macOS app that ran from its disk image or from Gatekeeper's App Translocation
mount. :func:`status` reports those as :attr:`Status.STALE`, :func:`enable`
refuses to register a macOS app from such a location, and :func:`refresh`
(called by the app at startup) re-points a stale entry at the running copy.

Filesystem locations and registry access go through small module-level helpers
(``_mac_plist_path``, ``_linux_desktop_path``, ``_reg_read``/``_reg_write_str``/
``_reg_delete``) so tests can redirect them.
"""

from __future__ import annotations

import logging
import os
import plistlib
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .. import APP_ID, APP_NAME, APP_SLUG, paths
from ..config import atomic_write_text

log = logging.getLogger(__name__)

__all__ = [
    "BACKGROUND_FLAG",
    "CONFIG_DIR_FLAG",
    "AutostartError",
    "Status",
    "disable",
    "enable",
    "is_enabled",
    "is_supported",
    "launch_command",
    "location",
    "refresh",
    "registered_command",
    "status",
]

BACKGROUND_FLAG = "--background"
CONFIG_DIR_FLAG = "--config-dir"
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
#: Key-file string escapes (applied before ``Exec`` quoting is interpreted).
_DESKTOP_STRING_ESCAPES = {"s": " ", "n": "\n", "t": "\t", "r": "\r", "\\": "\\"}

_TRANSIENT_LOCATION_MESSAGE = (
    f"{APP_NAME} is running from its disk image or from a temporary location, "
    "which will not exist at the next login. Move it to the Applications folder, "
    "open it from there, then turn on Start at login."
)
_LOGIN_ITEMS_MESSAGE = (
    f"macOS keeps {APP_NAME} from starting at login. Allow it in System Settings › "
    "General › Login Items (under “Allow in the Background”)."
)

#: launchd's command-line tool, by absolute path (an app may start with a bare PATH).
_LAUNCHCTL = "/bin/launchctl"
_LAUNCHCTL_TIMEOUT_S = 2.0
#: One line of ``launchctl print-disabled``: ``"<label>" => <state>``.
_LAUNCHD_ENTRY_RE = re.compile(r'^\s*"(?P<label>[^"]+)"\s*=>\s*(?P<state>\S+)\s*$')


class AutostartError(OSError):
    """Enabling or disabling start at login failed (message is user-presentable)."""


class Status(StrEnum):
    """State of the login entry, as seen from one ``--config-dir`` profile."""

    #: The entry starts this profile, and its program exists.
    ENABLED = "enabled"
    #: No entry, or one that was switched off (Task Manager, GNOME Tweaks, launchd).
    DISABLED = "disabled"
    #: The entry starts this profile, but its program is gone or sits in a
    #: temporary location (macOS disk image, App Translocation).
    STALE = "stale"
    #: The entry starts another ``--config-dir`` profile.
    OTHER_PROFILE = "other-profile"


@dataclass(frozen=True, slots=True)
class _Entry:
    """A login entry as found on disk or in the registry."""

    #: Its command line split into arguments (empty when there is none).
    command: list[str]
    #: ``False`` when the OS or desktop switched it off without deleting it.
    active: bool


# ------------------------------------------------------------------ public API
def is_supported() -> bool:
    """Whether this OS has an autostart implementation."""
    return _system() in {"windows", "macos", "linux"}


def launch_command(background: bool = True, config_dir: Path | None = None) -> list[str]:
    """Command line that starts this installation of the app.

    * frozen build inside an AppImage: the ``.AppImage`` file (the mounted
      bundle path changes on every run);
    * other frozen builds: the windowed executable (inside the ``.app`` bundle
      on macOS);
    * source/pip installs: ``python -m eye_tracker``, using ``pythonw.exe`` on
      Windows so no console window opens at login.

    A non-default profile (``config_dir``, see the module docs) adds
    ``--config-dir <dir>``. ``background`` appends ``--background`` (no
    first-run wizard or startup notifications), which is what a login launch
    wants.
    """
    command = _base_command()
    profile = _profile_dir(config_dir)
    if profile is not None:
        command += [CONFIG_DIR_FLAG, str(profile)]
    if background:
        command.append(BACKGROUND_FLAG)
    return command


def registered_command() -> list[str] | None:
    """The command of the login entry, split into arguments; ``None`` without one.

    Includes entries that were switched off in Task Manager or the desktop's
    settings. Never raises.
    """
    try:
        entry = _read_entry()
    except Exception:
        log.warning("Could not read the start-at-login entry", exc_info=True)
        return None
    return list(entry.command) if entry is not None and entry.command else None


def status(config_dir: Path | None = None) -> Status:
    """State of the login entry for the ``config_dir`` profile. Never raises."""
    try:
        entry = _read_entry()
    except Exception:
        log.warning("Could not read the start-at-login state", exc_info=True)
        return Status.DISABLED
    if entry is None or not entry.active or not entry.command:
        return Status.DISABLED
    if not _same_path(_command_profile(entry.command), _profile_dir(config_dir)):
        return Status.OTHER_PROFILE
    if not _program_available(entry.command[0]):
        return Status.STALE
    return Status.ENABLED


def is_enabled(config_dir: Path | None = None) -> bool:
    """Whether the app is registered to start the ``config_dir`` profile at login.

    ``False`` for an entry whose program no longer exists or that starts
    another profile (see :func:`status`). Never raises.
    """
    return status(config_dir) is Status.ENABLED


def enable(background: bool = True, config_dir: Path | None = None) -> None:
    """Register the app to start the ``config_dir`` profile at login.

    Replaces any previous entry, including another profile's. Raises
    :class:`AutostartError` on failure, on an unsupported OS, or on macOS when
    the app runs from its disk image or a translocated copy.
    """
    system = _system()
    try:
        command = launch_command(background, config_dir)
        if _is_transient_location(command[0]):
            raise AutostartError(_TRANSIENT_LOCATION_MESSAGE)
        if system == "windows":
            _win_enable(command)
        elif system == "macos":
            _mac_enable(command)
        elif system == "linux":
            _linux_enable(command)
        else:
            raise AutostartError(f"Start at login is not supported on {sys.platform}")
    except AutostartError:
        raise
    except Exception as exc:
        raise AutostartError(f"Could not enable start at login: {exc}") from exc
    log.info("Start at login enabled (%s)", location())


def disable(config_dir: Path | None = None) -> None:
    """Remove the login entry of the ``config_dir`` profile. A missing entry is not an error.

    An entry that starts another profile is left alone (and logged): it is not
    this profile's to remove. Raises :class:`AutostartError` when an existing
    entry cannot be removed.
    """
    system = _system()
    if system not in {"windows", "macos", "linux"}:
        return
    try:
        entry = _read_entry()
    except Exception:
        entry = None  # unreadable: removing it is the only sensible thing to do
    if entry is not None and entry.command:
        owner = _command_profile(entry.command)
        if not _same_path(owner, _profile_dir(config_dir)):
            log.info("Start at login starts another profile (%s); left alone", owner or "default")
            return
    try:
        if system == "windows":
            _win_disable()
        elif system == "macos":
            _unlink(_mac_plist_path())
        else:
            _unlink(_linux_desktop_path())
    except Exception as exc:
        raise AutostartError(f"Could not disable start at login: {exc}") from exc
    log.info("Start at login disabled")


def refresh(config_dir: Path | None = None) -> bool:
    """Point this profile's login entry at the running copy of the app, if it moved.

    Meant for app startup. The entry is rewritten when it is active, starts the
    ``config_dir`` profile and records a different command than
    :func:`launch_command` gives now, and

    * its program is gone or in a temporary location (a replaced AppImage, a
      moved folder, a macOS app first run from its disk image), or
    * this is a packaged build, which then replaces what the entry started
      (e.g. an older AppImage, or a source checkout's interpreter).

    It is never pointed at a temporary location, and a working entry is not
    taken over by a source run (a developer's checkout). With several packaged
    copies of one profile, the one started last wins. The ``--background``
    choice of the entry is kept. Returns ``True`` when the entry was rewritten.
    Never raises.
    """
    try:
        entry = _read_entry()
        if entry is None or not entry.active or not entry.command:
            return False
        profile = _profile_dir(config_dir)
        if not _same_path(_command_profile(entry.command), profile):
            return False
        background = BACKGROUND_FLAG in entry.command[1:]
        wanted = launch_command(background, profile)
        if _same_command(entry.command, wanted) or _is_transient_location(wanted[0]):
            return False
        if _program_available(entry.command[0]) and not paths.is_frozen():
            return False
        enable(background, profile)
        log.info("Start at login updated: %s (was %s)", wanted[0], entry.command[0])
    except Exception:
        log.warning("Could not update the start-at-login entry", exc_info=True)
        return False
    return True


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
    """The windowed executable, even when called from a console one.

    Registering a console build would open a terminal window at every login.
    The console builds are the ``*-cli`` executables and, on Windows, every
    other executable not named like the windowed one: the installer adds
    ``eye-tracker.exe`` (a copy of ``eye-tracker-cli.exe``) for the
    ``eye-tracker`` command on the PATH, so ``eye-tracker autostart enable``
    runs as that.
    """
    exe = Path(sys.executable)
    gui_names = {name.lower() for name in _GUI_EXECUTABLE_NAMES}
    console = exe.stem.lower().endswith("-cli") or (
        _system() == "windows" and exe.name.lower() not in gui_names
    )
    if console:
        for name in _GUI_EXECUTABLE_NAMES:
            candidate = exe.with_name(name)
            if candidate.name.lower() != exe.name.lower() and candidate.is_file():
                return candidate
    return exe


def _source_interpreter() -> str:
    exe = Path(sys.executable)
    if _system() == "windows" and exe.name.lower() == "python.exe":
        pythonw = exe.with_name("pythonw.exe")
        if pythonw.is_file():
            return str(pythonw)
    return str(exe)


def _active_profile() -> Path | None:
    """The ``--config-dir`` of this process; ``None`` for the default profile.

    ``paths`` keeps the override resolved (see ``paths.set_base_override``).
    """
    return paths.base_override()


def _profile_dir(config_dir: Path | str | None) -> Path | None:
    """The profile an entry is for: ``config_dir`` resolved like ``paths`` does,
    or the running process's profile when ``None``."""
    if config_dir is None:
        return _active_profile()
    return Path(config_dir).expanduser().resolve()


def _command_profile(command: Sequence[str]) -> str | None:
    """The ``--config-dir`` value in a command line, ``None`` for the default profile."""
    args = list(command[1:])
    for index, arg in enumerate(args):
        if arg == CONFIG_DIR_FLAG and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(CONFIG_DIR_FLAG + "="):
            return arg.partition("=")[2]
    return None


def _without_profile(command: Sequence[str]) -> list[str]:
    """The arguments of ``command`` (after the program) other than ``--config-dir``."""
    rest: list[str] = []
    args = iter(command[1:])
    for arg in args:
        if arg == CONFIG_DIR_FLAG:
            next(args, None)
        elif not arg.startswith(CONFIG_DIR_FLAG + "="):
            rest.append(arg)
    return rest


def _path_key(path: str | Path) -> str:
    """Comparable form of a path; case-insensitive for Windows entries."""
    text = os.path.normpath(os.path.expanduser(str(path)))
    return text.casefold() if _system() == "windows" else text


def _same_path(a: str | Path | None, b: str | Path | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return _path_key(a) == _path_key(b)


def _same_command(recorded: Sequence[str], wanted: Sequence[str]) -> bool:
    """Whether two commands start the same program with the same profile and options."""
    return (
        bool(recorded)
        and bool(wanted)
        and _same_path(recorded[0], wanted[0])
        and _same_path(_command_profile(recorded), _command_profile(wanted))
        and _without_profile(recorded) == _without_profile(wanted)
    )


def _program_available(program: str) -> bool:
    """Whether an entry's program will still be there at the next login."""
    if not program or _is_transient_location(program):
        return False
    if _system() == "windows":
        program = os.path.expandvars(program)  # a hand-made REG_EXPAND_SZ entry
    if os.path.isabs(program):
        return os.path.exists(program)
    return shutil.which(program) is not None


def _is_transient_location(program: str) -> bool:
    """Whether ``program`` runs from a place that will be gone at the next login (macOS).

    Gatekeeper's App Translocation runs a quarantined app from a random mount
    under ``/private/var/folders/.../AppTranslocation/``, and an app opened
    straight from its disk image lives on a read-only ``/Volumes/`` mount that
    disappears when the image is ejected. (Apps on an external drive also live
    under ``/Volumes/``, but on a writable volume.)
    """
    if _system() != "macos":
        return False
    if "/AppTranslocation/" in program:
        return True
    return program.startswith("/Volumes/") and _is_read_only_volume(program)


def _is_read_only_volume(path: str) -> bool:
    """Whether ``path`` is on a read-only filesystem (a mounted disk image)."""
    statvfs = getattr(os, "statvfs", None)
    if statvfs is None:
        return False
    try:
        flags = statvfs(path).f_flag
    except OSError:
        return False
    return bool(flags & getattr(os, "ST_RDONLY", 1))


def _read_entry() -> _Entry | None:
    """The login entry of the current OS, ``None`` when there is none."""
    system = _system()
    if system == "windows":
        return _win_entry()
    if system == "macos":
        return _mac_entry()
    if system == "linux":
        return _linux_entry()
    return None


# -------------------------------------------------------------------- Windows
def _win_command_line(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def _win_split(command_line: str) -> list[str]:
    """Split a command line the way ``CommandLineToArgvW`` and the C runtime do.

    The program name ends at the next space, or at the closing quote when it is
    quoted; backslashes mean nothing there. For the other arguments, ``2n``
    backslashes before a quote become ``n`` and the quote delimits, ``2n+1``
    become ``n`` and a literal quote, and ``""`` inside quotes is a literal
    quote. This is the inverse of ``subprocess.list2cmdline``.
    """
    text = command_line.lstrip(" \t")
    if not text:
        return []
    if text.startswith('"'):
        end = text.find('"', 1)
        end = len(text) if end < 0 else end
        args = [text[1:end]]
        index = end + 1
    else:
        end = len(text)
        for position, char in enumerate(text):
            if char in " \t":
                end = position
                break
        args = [text[:end]]
        index = end
    current: list[str] = []
    in_quotes = started = False
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\":
            run = index
            while run < length and text[run] == "\\":
                run += 1
            count = run - index
            if run < length and text[run] == '"':
                current.append("\\" * (count // 2))
                if count % 2:
                    current.append('"')
                    index = run + 1
                else:
                    index = run  # the quote is a delimiter, handled next
            else:
                current.append("\\" * count)
                index = run
            started = True
            continue
        if char == '"':
            if in_quotes and index + 1 < length and text[index + 1] == '"':
                current.append('"')
                index += 2
                continue
            in_quotes = not in_quotes
            started = True
        elif char in " \t" and not in_quotes:
            if started:
                args.append("".join(current))
                current, started = [], False
        else:
            current.append(char)
            started = True
        index += 1
    if started:
        args.append("".join(current))
    return args


def _win_entry() -> _Entry | None:
    value = _reg_read(RUN_KEY, RUN_VALUE_NAME)
    if not isinstance(value, str) or not value.strip():
        return None
    return _Entry(_win_split(value), active=not _win_disabled_in_task_manager())


def _win_disabled_in_task_manager() -> bool:
    """Task Manager keeps the Run value but flags it; the first byte is odd when disabled."""
    data = _reg_read(STARTUP_APPROVED_KEY, RUN_VALUE_NAME)
    return isinstance(data, (bytes, bytearray)) and len(data) > 0 and data[0] & 1 == 1


def _win_enable(command: Sequence[str]) -> None:
    _reg_write_str(RUN_KEY, RUN_VALUE_NAME, _win_command_line(command))
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


def _mac_enable(command: Sequence[str]) -> None:
    _write_text(_mac_plist_path(), _mac_plist(command).decode("utf-8"))
    if not _mac_disabled_in_launchd():
        return
    # Switched off in launchd's records, which outlive the plist: without this
    # the checkbox would silently do nothing. ``enable`` only clears that
    # record; it does not load the agent, which still starts at the next login.
    domain = _mac_launchd_domain()
    if domain is not None:
        _launchctl("enable", f"{domain}/{LAUNCH_AGENT_LABEL}")
    if _mac_disabled_in_launchd():
        raise AutostartError(_LOGIN_ITEMS_MESSAGE)


def _mac_disabled_in_launchd() -> bool:
    """Whether launchd itself keeps the agent from starting, whatever the plist says.

    ``launchctl disable`` (and ``unload -w``) record a switch-off in launchd's
    per-user database and leave the plist untouched; the Login Items switch in
    System Settings may do so as well. ``False`` when that cannot be read.
    """
    domain = _mac_launchd_domain()
    if domain is None:
        return False
    listing = _launchctl("print-disabled", domain)
    return listing is not None and _launchd_lists_disabled(listing, LAUNCH_AGENT_LABEL)


def _launchd_lists_disabled(listing: str, label: str) -> bool:
    """Whether ``launchctl print-disabled`` output switches ``label`` off.

    Its lines read ``"<label>" => disabled`` (``=> true`` on older macOS);
    other sections of the output map labels to app identifiers instead.
    """
    for line in listing.splitlines():
        match = _LAUNCHD_ENTRY_RE.match(line)
        if match and match["label"] == label and match["state"].lower() in {"disabled", "true"}:
            return True
    return False


def _mac_launchd_domain() -> str | None:
    """launchd's domain for this user's login session (``gui/<uid>``); ``None`` off POSIX."""
    getuid = getattr(os, "getuid", None)
    return f"gui/{getuid()}" if getuid is not None else None


def _launchctl(*args: str) -> str | None:
    """Run ``launchctl`` and return its output; ``None`` when it failed (tests replace it)."""
    argv = [_LAUNCHCTL, *args]
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_LAUNCHCTL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        log.debug("%s failed: %s", " ".join(argv), exc)
        return None
    if result.returncode != 0:
        log.debug(
            "%s exited with %s: %s",
            " ".join(argv),
            result.returncode,
            (result.stderr or "").strip()[:200],
        )
        return None
    return result.stdout or ""


def _mac_entry() -> _Entry | None:
    path = _mac_plist_path()
    try:
        agent = plistlib.loads(path.read_bytes())
    except FileNotFoundError:
        return None
    except Exception:
        log.debug("Unreadable LaunchAgent %s", path, exc_info=True)
        return _Entry([], active=False)  # launchd would ignore it as well
    if not isinstance(agent, dict):
        return _Entry([], active=False)
    arguments = agent.get("ProgramArguments")
    command = (
        [str(arg) for arg in arguments]
        if isinstance(arguments, list) and all(isinstance(a, str) for a in arguments)
        else []
    )
    active = agent.get("Disabled") is not True
    if active and command and _mac_disabled_in_launchd():
        active = False  # launchd will not start it at login (see _mac_disabled_in_launchd)
    return _Entry(command, active=active)


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


def _desktop_exec_split(value: str) -> list[str]:
    """Split an ``Exec`` value into arguments (the inverse of :func:`_desktop_exec`).

    Undoes the key-file string escapes, then the ``Exec`` quoting (only double
    quotes; ``\\"``, ``\\```, ``\\$`` and ``\\\\`` inside them), then ``%%``.
    """
    unescaped: list[str] = []
    chars = iter(value)
    for char in chars:
        if char == "\\":
            following = next(chars, "")
            unescaped.append(_DESKTOP_STRING_ESCAPES.get(following, "\\" + following))
        else:
            unescaped.append(char)
    text = "".join(unescaped)
    args: list[str] = []
    current: list[str] = []
    in_quotes = started = False
    index = 0
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if in_quotes:
            if char == "\\" and following and following in _DESKTOP_QUOTED_ESCAPES:
                current.append(following)
                index += 1
            elif char == '"':
                in_quotes = False
            else:
                current.append(char)
        elif char == '"':
            in_quotes = started = True
        elif char in " \t":
            if started:
                args.append("".join(current))
                current, started = [], False
        else:
            current.append(char)
            started = True
        index += 1
    if started:
        args.append("".join(current))
    return [arg.replace("%%", "%") for arg in args]


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


def _linux_enable(command: Sequence[str]) -> None:
    _write_text(_linux_desktop_path(), _desktop_entry(command))


def _linux_entry() -> _Entry | None:
    path = _linux_desktop_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    entry = _parse_desktop_entry(text)
    # Desktop environments "turn off" an autostart entry by editing these keys
    # (GNOME Tweaks, KDE System Settings) rather than deleting the file.
    active = (
        entry.get("Hidden", "false").lower() != "true"
        and entry.get("X-GNOME-Autostart-enabled", "true").lower() != "false"
    )
    exec_value = entry.get("Exec", "")
    return _Entry(_desktop_exec_split(exec_value) if exec_value else [], active=active)


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
