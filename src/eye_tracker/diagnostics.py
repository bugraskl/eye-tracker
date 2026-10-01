"""``eye-tracker doctor`` and ``eye-tracker bench``.

:func:`collect_report` gathers everything useful for a bug report into a
JSON-friendly dict and :func:`format_report` renders it for humans. Collection
never fails as a whole: a section that cannot be gathered carries an ``error``
entry instead. The report is meant to be pasted into public bug reports, so it
must not reveal the user's account name: the home directory is shown as ``~``
wherever it appears (paths, quoted paths, error messages), camera files by
their name only, camera device links without their serial number, and nothing
derived from the user name (such as the instance socket name) is included.

Whether the global hotkeys could be registered is only known to the app that
registered them: the settings window passes its hotkey manager in, and the
command-line ``doctor`` asks a running instance (``status``).

:func:`run_bench` measures what the vision pipeline costs on this machine, once
as fast as possible and once at the rate the app uses while you sit still.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
import platform as py_platform
import re
import shlex
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from . import APP_NAME, __version__, paths
from .cli import FROZEN_CLI_NAME, cli_command, format_command
from .config import Settings, describe_settings
from .platform.base import PlatformServices
from .types import Monitor, Rect, layout_signature, virtual_bounds

if TYPE_CHECKING:
    from .platform.hotkeys import HotkeyManager

__all__ = [
    "BenchError",
    "collect_report",
    "current_monitors",
    "format_bench",
    "format_command",
    "format_report",
    "run_bench",
]

log = logging.getLogger(__name__)

_DISTRIBUTIONS = {
    "numpy": ("numpy",),
    "opencv": (
        "opencv-python-headless",
        "opencv-python",
        "opencv-contrib-python-headless",
        "opencv-contrib-python",
    ),
    "PySide6": ("PySide6-Essentials", "PySide6"),
    "psutil": ("psutil",),
    "platformdirs": ("platformdirs",),
}

#: Distributions Eye Tracker never uses that are a problem when installed next to
#: it, with the reason. The MediaPipe runtime is not needed (its models run in
#: OpenCV) and ships a usage logger (telemetry), which the offline rule excludes.
_UNWANTED_DISTRIBUTIONS: dict[str, str] = {
    "mediapipe": "Eye Tracker does not use it, and it contains a usage logger (telemetry)",
}

#: Things worth knowing on Linux that are not problems (shown under "OS integration").
_LINUX_CAMERA_NOTE = (
    "apps that open /dev/video* are noticed (refused attempts too, via fanotify on "
    "Linux 5.13+); apps using the PipeWire camera portal are not: add them to "
    "'Pause while these apps run' (e.g. zoom, teams-for-linux, skypeforlinux, obs)"
)
_WAYLAND_CURSOR_NOTE = (
    "moved exactly with swaymsg (sway) or hyprctl (Hyprland); elsewhere it needs "
    "ydotool 1.x with ydotoold running and a flat pointer-acceleration profile for "
    "ydotool's virtual device (ydotool 0.1 is not supported)"
)
#: ``eye-tracker-cli`` on macOS: the system judges the Accessibility permission of
#: whatever started the command (the terminal), so the app's own grant is unknown.
_MAC_CLI_ACCESSIBILITY_NOTE = (
    "unknown from the command line: macOS judges the Accessibility permission of the "
    "program that started this command (the terminal), not the app's own. The app's "
    "Settings > Diagnostics shows the app's state"
)
#: A lock tool being installed does not mean it locks this session: loginctl is
#: on every systemd system, but a bare i3/sway session has nothing listening.
_LINUX_LOCK_NOTE = (
    "only shows which lock tools are installed; whether one locks this session is "
    "known when it is tried. If nothing locks, you get a 'Could not lock the screen' "
    "notification: test it once with a short 'After ... s away'"
)
#: ``hotkeys.registration`` when no running app can say what the OS accepted.
_HOTKEYS_NOT_RUNNING = (
    "unknown: Eye Tracker is not running. When it starts, a hotkey the system refuses "
    "(for example because another app uses it) is reported in a 'Hotkey unavailable' "
    "notification and in the log"
)
_HOTKEYS_NOT_REPORTED = (
    "unknown: the running app does not report it; see its 'Hotkey unavailable' "
    "notification or the log"
)
#: How long ``doctor`` waits for the running instance's ``status``.
_STATUS_TIMEOUT_MS = 1000
#: The serial number at the end of a udev ``/dev/v4l/by-id`` name
#: (``usb-<vendor>_<model>_<serial>-video-index0``).
_BY_ID_SERIAL = re.compile(r"_[^_/]+(-video-index\d+)$")

#: Shown for empty values. Reports are pasted into bug reports and printed on
#: consoles with legacy code pages, so the text output is kept ASCII-only.
_EMPTY = "-"
#: Qt platform plugins without real screens (their "monitor" is a fake).
_VIRTUAL_PLATFORMS = frozenset({"offscreen", "minimal"})


class _State:
    """A Qt application created here for command-line use (kept alive), and
    whether it had to fall back to the offscreen platform for lack of a display."""

    owned_app: ClassVar[list[Any]] = []
    no_display: ClassVar[bool] = False


# ======================================================================= doctor
def collect_report(
    probe_cameras: bool = False, *, hotkey_manager: HotkeyManager | None = None
) -> dict[str, Any]:
    """Gather diagnostics. Creates a ``QGuiApplication`` if none exists (for monitors).

    Args:
        probe_cameras: Briefly open camera indices 0-3 (their lights may flash).
        hotkey_manager: The app's own hotkey manager (the settings window passes
            it), which knows what the OS accepted. Without it, a running
            instance is asked for its ``status``.
    """
    services = _platform_services()
    settings, settings_info = _settings_section()
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "app": _safe(_app_section),
        "system": _safe(lambda: _system_section(services)),
        "qt": _safe(_qt_section),
        "monitors": _safe(_monitors_section),
        "libraries": _safe(_libraries_section),
        "backends": _safe(lambda: _backends_section(settings)),
        "cameras": _safe(lambda: _cameras_section(settings, probe_cameras)),
        "platform": _safe(lambda: _platform_section(services)),
    }
    running = _instance_running(report)
    report["hotkeys"] = _safe(lambda: _hotkeys_section(settings, hotkey_manager, running))
    report["autostart"] = _safe(_autostart_section)
    report["paths"] = _safe(_paths_section)
    report["settings"] = settings_info
    report["calibration"] = _safe(lambda: _calibration_section(report, settings))
    report["problems"] = _problems(report)
    # Every section redacts its paths; this catches what they cannot foresee
    # (an OS error quoting a file under the home directory, a note, ...).
    redacted: dict[str, Any] = _redact_tree(report)
    return redacted


def format_report(report: dict[str, Any]) -> str:
    """Render a :func:`collect_report` dict as aligned, human-readable text."""
    app = report.get("app", {})
    title = f"{APP_NAME} {app.get('version', __version__)} - diagnostics"
    lines = [title, "=" * len(title), f"Generated {report.get('generated_at', '?')}", ""]

    problems = report.get("problems") or []
    lines.append("Problems")
    if problems:
        lines.extend(f"  ! {p}" for p in problems)
    else:
        lines.append("  none found")
    lines.append("")

    for key, heading in (
        ("app", "App"),
        ("system", "System"),
        ("qt", "Qt"),
        ("libraries", "Libraries"),
        ("backends", "Vision backends"),
        ("cameras", "Cameras"),
        ("monitors", "Monitors"),
        ("platform", "OS integration"),
        ("hotkeys", "Hotkeys"),
        ("autostart", "Start at login"),
        ("calibration", "Calibration"),
        ("settings", "Settings"),
        ("paths", "Paths"),
    ):
        section = report.get(key)
        if section is None:
            continue
        lines.append(heading)
        lines.extend(_format_section(key, section))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def current_monitors() -> list[Monitor]:
    """Monitors as Qt sees them (Qt global coordinates). Needs a ``QGuiApplication``."""
    from PySide6.QtGui import QGuiApplication

    screens = QGuiApplication.screens()
    primary = QGuiApplication.primaryScreen()
    monitors = []
    for index, screen in enumerate(screens):
        geo = screen.geometry()
        monitors.append(
            Monitor(
                index=index,
                name=screen.name() or f"Screen {index + 1}",
                rect=Rect(geo.x(), geo.y(), geo.width(), geo.height()),
                primary=screen == primary,
                scale=float(screen.devicePixelRatio()),
            )
        )
    return monitors


# ------------------------------------------------------------------ sections
def _safe(collect: Callable[[], Any]) -> Any:
    try:
        return collect()
    except Exception as exc:
        log.debug("Diagnostics section failed", exc_info=True)
        return {"error": _error_text(exc)}


def _error_text(exc: BaseException) -> str:
    """``"PermissionError: …"`` without the home directory (OS errors quote paths)."""
    return _redact(f"{type(exc).__name__}: {exc}") or type(exc).__name__


def _platform_services() -> PlatformServices:
    try:
        from .platform import get_platform

        return get_platform()
    except Exception:
        log.debug("Platform services unavailable", exc_info=True)
        return PlatformServices()


def _app_section() -> dict[str, Any]:
    from . import ipc

    running: bool | None
    try:
        _ensure_gui_app()
        running = ipc.is_running()
    except Exception:
        running = None
    return {
        "name": APP_NAME,
        "version": __version__,
        "frozen": paths.is_frozen(),
        "executable": _redact(sys.executable),
        # What "eye-tracker ..." means for this copy (release packages do not
        # put an eye-tracker command on the PATH).
        "command_line": _redacted_command(cli_command()),
        "python": py_platform.python_version(),
        "implementation": py_platform.python_implementation(),
        "instance_running": running,
    }


def _system_section(services: PlatformServices) -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": py_platform.system(),
        "release": py_platform.release(),
        "version": py_platform.version(),
        "machine": py_platform.machine(),
        "platform": py_platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        info["macos_version"] = py_platform.mac_ver()[0]
    if sys.platform.startswith("linux"):
        info["session_type"] = os.environ.get("XDG_SESSION_TYPE")
        info["desktop"] = os.environ.get("XDG_CURRENT_DESKTOP")
        info["display"] = bool(os.environ.get("DISPLAY"))
        info["wayland_display"] = bool(os.environ.get("WAYLAND_DISPLAY"))
    info["wayland"] = bool(services.is_wayland)
    return info


def _qt_section() -> dict[str, Any]:
    import PySide6
    from PySide6.QtCore import qVersion

    app = _ensure_gui_app()
    return {
        "version": qVersion(),
        "pyside": PySide6.__version__,
        "platform_plugin": app.platformName() if app is not None else None,
        "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "not set"),
        "QT_ENABLE_HIGHDPI_SCALING": os.environ.get("QT_ENABLE_HIGHDPI_SCALING", "not set"),
    }


def _monitors_section() -> dict[str, Any]:
    app = _ensure_gui_app()
    if app is None:
        return {"error": "no Qt GUI application available"}
    if _State.no_display:
        return {"error": "no display available (DISPLAY and WAYLAND_DISPLAY are not set)"}
    from PySide6.QtGui import QGuiApplication

    monitors = current_monitors()
    screens = QGuiApplication.screens()
    items = []
    for monitor, screen in zip(monitors, screens, strict=False):
        items.append(
            {
                "index": monitor.index,
                "name": monitor.name,
                "rect": monitor.rect.to_list(),
                "primary": monitor.primary,
                "scale": monitor.scale,
                "logical_dpi": round(float(screen.logicalDotsPerInch()), 1),
                "refresh_hz": round(float(screen.refreshRate()), 1),
            }
        )
    out: dict[str, Any] = {"count": len(items), "items": items}
    if app.platformName() in _VIRTUAL_PLATFORMS:
        # The offscreen plugin invents one screen; do not report it as real.
        out["virtual"] = True
    if monitors:
        out["virtual_desktop"] = virtual_bounds(monitors).to_list()
        out["layout_signature"] = layout_signature(monitors)
    return out


def _libraries_section() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for label, names in _DISTRIBUTIONS.items():
        versions[label] = None
        for name in names:
            try:
                versions[label] = importlib.metadata.version(name)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    unwanted = _unwanted_distributions()
    if unwanted:
        versions["unwanted"] = unwanted
    return versions


def _unwanted_distributions() -> dict[str, str]:
    """``{name: version}`` of the installed :data:`_UNWANTED_DISTRIBUTIONS`."""
    found: dict[str, str] = {}
    for name in _UNWANTED_DISTRIBUTIONS:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:  # a broken installation record still means "installed"
            log.debug("Reading the version of %s failed", name, exc_info=True)
            found[name] = "unknown version"
    return found


def _backends_section(settings: Settings) -> dict[str, Any]:
    from .vision.backends import (
        BACKEND_MODEL_FILES,
        MODEL_FILES,
        BackendUnavailable,
        available_backends,
        backend_class,
    )

    models = {}
    for filename, digest in MODEL_FILES.items():
        path = paths.model_path(filename)
        users = [name for name, files in BACKEND_MODEL_FILES.items() if filename in files]
        entry: dict[str, Any] = {"used_by": users, "present": path.is_file()}
        if entry["present"]:
            entry["size"] = path.stat().st_size
            entry["sha256_ok"] = _sha256(path) == digest
        models[filename] = entry
    out: dict[str, Any] = {
        "configured": settings.general.backend,
        "available": available_backends(),
        "models": models,
    }
    try:
        cls = backend_class(settings.general.backend)
        out["active"] = cls.name
        out["feature_version"] = cls.feature_version
    except BackendUnavailable as exc:
        out["active"] = None
        out["active_error"] = _redact(str(exc))
    return out


def _cameras_section(settings: Settings, probe: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "configured": _describe_device(settings.camera.device),
        "resolution": [settings.camera.width, settings.camera.height],
        "api": settings.camera.api,
        "probed": probe,
    }
    if probe:
        from .vision.camera import list_cameras

        out["devices"] = [
            {"index": c.index, "name": c.name, "width": c.width, "height": c.height}
            for c in list_cameras(max_index=4, api=settings.camera.api)
        ]
    return out


def _platform_section(services: PlatformServices) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": services.name,
        "wayland": bool(services.is_wayland),
        "cursor_position_reliable": bool(services.cursor_position_reliable()),
        "capabilities": dict(services.capabilities()),
        "permissions": dict(services.permissions()),
    }
    accessibility = _accessibility_status(services)
    if accessibility != "unknown":
        out["accessibility"] = accessibility
    lock_methods = _lock_methods(services)
    if lock_methods is not None:
        out["lock_methods"] = lock_methods
    notes: dict[str, str] = {}
    if sys.platform.startswith("linux"):
        notes["camera_release"] = _LINUX_CAMERA_NOTE
        if services.is_wayland:
            notes["cursor"] = _WAYLAND_CURSOR_NOTE
        notes["lock"] = _LINUX_LOCK_NOTE
    if sys.platform == "darwin" and out["permissions"].get("accessibility") is None and _is_cli():
        notes["accessibility"] = _MAC_CLI_ACCESSIBILITY_NOTE
    if notes:
        out["notes"] = notes
    return out


def _is_cli() -> bool:
    """Whether this is the console executable of a release package (``eye-tracker-cli``)."""
    return paths.is_frozen() and Path(sys.executable).stem.lower() == FROZEN_CLI_NAME


def _accessibility_status(services: PlatformServices) -> str:
    """``accessibility_status()`` (only macOS knows more than ``"unknown"``)."""
    try:
        return str(services.accessibility_status())
    except Exception:
        log.debug("accessibility_status() failed", exc_info=True)
        return "unknown"


def _lock_methods(services: PlatformServices) -> list[str] | None:
    """The screen-lock methods available here, in the order they are tried.

    Read from ``services.lock_methods()`` where the platform provides it;
    ``None`` otherwise (the ``lock`` capability is all that is known then).
    """
    getter = getattr(services, "lock_methods", None)
    if not callable(getter):
        return None
    try:
        return [str(method) for method in getter()]
    except Exception:
        log.debug("lock_methods() failed", exc_info=True)
        return None


#: ``settings.hotkeys`` fields, in the order the settings window shows them.
_HOTKEY_NAMES = ("toggle_tracking", "toggle_privacy", "recalibrate")


def _hotkeys_section(
    settings: Settings, live_manager: HotkeyManager | None, running: bool
) -> dict[str, Any]:
    """Configured hotkeys, what is wrong with them, and what the OS accepted.

    ``doctor`` cannot register anything itself (a running app holds its
    hotkeys, and grabbing them would change global state), so registration
    results come from ``live_manager`` (the app's own manager) or a running
    instance; otherwise they are unknown.
    """
    from .platform.hotkeys import create_hotkey_manager, parse_hotkey

    manager = create_hotkey_manager()
    configured: dict[str, Any] = {}
    invalid: dict[str, str] = {}
    conflicts: dict[str, str] = {}
    for name in _HOTKEY_NAMES:
        text = getattr(settings.hotkeys, name)
        configured[name] = text
        if not text:
            continue
        try:
            hotkey = parse_hotkey(text)
        except ValueError as exc:
            invalid[name] = str(exc)
            continue
        # Read-only: e.g. Ctrl+Alt+T is AltGr+T on some Windows layouts, and the
        # manager refuses to register a combination that types a character.
        try:
            conflict = manager.layout_conflict(hotkey)
        except Exception:
            log.debug("layout_conflict(%s) failed", text, exc_info=True)
            conflict = None
        if conflict:
            conflicts[name] = conflict
    # The manager that runs knows more than a fresh one (e.g. that another
    # program grabs the Super key on its own, which only shows once grabbing).
    instance = _instance_hotkeys() if live_manager is None and running else None
    if live_manager is not None:
        note = live_manager.note or ""
    elif instance is not None and instance[2]:
        note = instance[2]
    else:
        note = manager.note or ""
    out: dict[str, Any] = {
        "enabled": settings.hotkeys.enabled,
        "backend": manager.name,
        "supported": bool(manager.supported),
        "note": note,
        "configured": configured,
        "invalid": invalid,
        "layout_conflicts": conflicts,
    }
    wanted = [name for name in _HOTKEY_NAMES if configured[name] and name not in invalid]
    if settings.hotkeys.enabled and wanted and manager.supported:
        out["registration"] = _hotkey_registration(
            wanted, live_manager, running, instance=instance if running else _ASK
        )
    return out


#: Default of ``_hotkey_registration(instance=...)``: ask the running instance.
_ASK: Any = object()


def _hotkey_registration(
    names: Sequence[str],
    live_manager: HotkeyManager | None,
    running: bool,
    *,
    instance: tuple[set[str], dict[str, str], str] | None = _ASK,
) -> dict[str, str] | str:
    """``{name: "registered" | "not registered: <why>"}``, or why that is unknown.

    ``instance`` is what :func:`_instance_hotkeys` returned, when the caller
    asked already.
    """
    if live_manager is not None:
        registered = set(live_manager.registered)
        errors: dict[str, str] = {}
        for name in names:
            try:
                reason = live_manager.last_error(name)
            except Exception:
                log.debug("last_error(%s) failed", name, exc_info=True)
                reason = None
            if reason:
                errors[name] = str(reason)
    elif not running:
        return _HOTKEYS_NOT_RUNNING
    else:
        live = _instance_hotkeys() if instance is _ASK else instance
        if live is None:
            return _HOTKEYS_NOT_REPORTED
        registered, errors, _note = live
    return {
        name: "registered"
        if name in registered
        else f"not registered: {errors.get(name) or 'reason unknown'}"
        for name in names
    }


def _instance_hotkeys() -> tuple[set[str], dict[str, str], str] | None:
    """What the running instance's ``status`` says about its hotkeys.

    ``(registered names, {name: why it is not registered}, its manager's note)``;
    ``None`` when the instance does not answer or does not report its hotkeys
    (an older version).
    """
    status = _instance_status()
    hotkeys = status.get("hotkeys") if status is not None else None
    if not isinstance(hotkeys, Mapping):
        return None
    registered = hotkeys.get("registered")
    # A list of names, or a {name: hotkey} mapping: iterating gives the names.
    names = {str(n) for n in registered} if isinstance(registered, (list, Mapping)) else set()
    raw_errors = hotkeys.get("errors")
    errors = (
        {str(k): str(v) for k, v in raw_errors.items() if v}
        if isinstance(raw_errors, Mapping)
        else {}
    )
    note = hotkeys.get("note")
    return names, errors, note if isinstance(note, str) else ""


def _instance_status() -> dict[str, Any] | None:
    """The running instance's ``status`` (read-only), or ``None``."""
    from . import ipc

    try:
        _ensure_gui_app()  # before ipc creates a plain QCoreApplication
        reply = ipc.send_command("status", timeout_ms=_STATUS_TIMEOUT_MS)
    except Exception:
        log.debug("Asking the running instance for its status failed", exc_info=True)
        return None
    if not reply or reply.startswith("error"):
        return None
    try:
        data = json.loads(reply)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _autostart_section() -> dict[str, Any]:
    from .platform import autostart

    registered = autostart.registered_command()
    return {
        "supported": autostart.is_supported(),
        "status": autostart.status().value,
        "enabled": autostart.is_enabled(),
        "location": _redact(autostart.location()),
        "registered": _redacted_command(registered) if registered else None,
        "command": _redacted_command(autostart.launch_command()),
    }


def _redacted_command(parts: Sequence[str], *, posix_shell: bool = os.name != "nt") -> str:
    """:func:`format_command` with the home directory shown as ``~``.

    A POSIX shell expands ``~`` only outside quotes, so there the ``~/`` that
    replaced the home directory at the start of an argument stays in front of
    the quoted rest, and a command the report suggests still runs when pasted:
    ``~/'my apps/eye-tracker'``, not ``'~/my apps/eye-tracker'``. Windows
    shells give ``~`` no meaning (there it only shortens the path), so
    ``posix_shell`` is ``False`` there and the command is rendered as
    :func:`format_command` renders it.
    """
    shown = [_redact(part) or "" for part in parts]
    if not posix_shell:
        return format_command(shown)
    pattern = _home_pattern()
    return " ".join(
        _shell_home_path(text) if pattern is not None and pattern.match(part) else shlex.quote(text)
        for part, text in zip(parts, shown, strict=True)
    )


def _shell_home_path(text: str) -> str:
    """``~`` or ``~/rest`` for a POSIX shell: the tilde bare, the rest quoted as needed."""
    tilde, slash, rest = text.partition("/")
    if tilde != "~":  # the home directory followed by something else than a slash
        return shlex.quote(text)
    return f"~/{shlex.quote(rest)}" if rest else tilde + slash


def _paths_section() -> dict[str, Any]:
    # No socket name: it is derived from the user name and would reveal it.
    return {
        "profile": "custom (--config-dir)" if _profile_override() else "default",
        "config_dir": _redact(paths.config_dir()),
        "data_dir": _redact(paths.data_dir()),
        "log_dir": _redact(paths.log_dir()),
        "settings_file": _redact(paths.settings_file()),
        "calibration_file": _redact(paths.calibration_file()),
        "log_file": _redact(paths.log_file()),
    }


def _profile_override() -> bool:
    """Whether a ``--config-dir`` profile is in use."""
    return paths.base_override() is not None


def _settings_section() -> tuple[Settings, dict[str, Any]]:
    """Settings as the app would load them, without the side effects of ``Settings.load``
    (which moves a corrupt file aside)."""
    path = paths.settings_file()
    info: dict[str, Any] = {"file_exists": path.is_file()}
    settings = Settings()
    if info["file_exists"]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            info["valid"] = False
            info["error"] = _error_text(exc)
        else:
            settings = Settings.from_dict(data)
            info["valid"] = isinstance(data, dict)
    defaults = Settings()
    info["non_default"] = {
        row["key"]: _shareable_setting(row["key"], _get_setting(settings, row["key"]))
        for row in describe_settings()
        if _get_setting(settings, row["key"]) != _get_setting(defaults, row["key"])
    }
    return settings, info


def _shareable_setting(key: str, value: Any) -> Any:
    """A setting value as it may appear in a pasted report (no account name)."""
    if key == "camera.device":
        return _describe_device(str(value))  # may be a video file's full path
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, list):
        return [_redact(v) if isinstance(v, str) else v for v in value]
    return value


def _calibration_section(report: dict[str, Any], settings: Settings) -> dict[str, Any]:
    from .gaze.store import CalibrationLibrary

    path = paths.calibration_file()
    if not path.is_file():
        return {"exists": False}
    library = CalibrationLibrary.load(path)
    data = library.latest
    if data is None:
        return {"exists": True, "valid": False}
    out: dict[str, Any] = {
        "exists": True,
        "valid": True,
        "profiles": len(library),
        "created_at": data.created_at,
        "backend": data.backend,
        "feature_version": data.feature_version,
        "camera": _describe_device(data.camera) if data.camera else None,
        "frame_size": list(data.frame_size) if all(data.frame_size) else None,
        "grade": data.grade,
        "monitors": len(data.monitors),
        "samples": len(data.samples),
        "implicit_samples": len(data.implicit_samples),
        "model_degree": getattr(data.model, "degree", None),
    }
    summary = _calibration_summary(data.report)
    if summary:
        out["summary"] = summary
    if len(library) > 1:
        # Most recently used first, one per desk / camera / backend.
        out["profile_list"] = {
            str(n): _describe_profile(profile)
            for n, profile in enumerate(library.profiles, start=1)
        }
    backends = report.get("backends") or {}
    monitors_info = report.get("monitors") or {}
    active = backends.get("active")
    feature_version = backends.get("feature_version")
    if active and feature_version and "items" in monitors_info:
        monitors = [
            Monitor(
                index=m["index"],
                name=m["name"],
                rect=Rect.from_list(m["rect"]),
                primary=m["primary"],
                scale=m["scale"],
            )
            for m in monitors_info["items"]
        ]
        # What the app does: the best profile for this setup, not just the latest.
        match, reason = library.match(
            active, feature_version, monitors, camera=settings.camera.device
        )
        out["compatible"] = match is not None
        if match is None:
            out["reason"] = reason
        elif match is not data:
            out["matching_profile"] = _describe_profile(match)
    else:
        out["compatible"] = None
    return out


def _describe_profile(data: Any) -> str:
    """One line per stored calibration, e.g. ``2026-09-30 facemesh, 2 monitors, camera 0, good``."""
    parts = [f"{str(data.created_at)[:10]} {data.backend}", f"{len(data.monitors)} monitor(s)"]
    if data.camera:
        parts.append(_describe_device(data.camera))
    if data.grade:
        parts.append(str(data.grade))
    return ", ".join(parts)


def _calibration_summary(report: dict[str, Any]) -> str | None:
    if not report:
        return None
    try:
        from .gaze.calibration import CalibrationReport

        return CalibrationReport.from_dict(report).summary()
    except Exception:
        return None


def _problems(report: dict[str, Any]) -> list[str]:
    """Plain-language list of things that stop features from working."""
    problems: list[str] = []
    for key, section in report.items():
        if isinstance(section, dict) and "error" in section and len(section) == 1:
            problems.append(f"Could not collect {key} information: {section['error']}")

    libraries = report.get("libraries") or {}
    for name, version in (libraries.get("unwanted") or {}).items():
        why = _UNWANTED_DISTRIBUTIONS.get(name, "Eye Tracker does not use it")
        problems.append(
            f"The {name} package ({version}) is installed in this Python environment. "
            f"{why}; uninstall it (pip uninstall {name})."
        )

    backends = report.get("backends") or {}
    if backends.get("available") == []:
        problems.append(
            "No vision backend is available (it needs OpenCV 4.10 or newer with its DNN "
            "module, and the bundled model files)."
        )
    for filename, model in (backends.get("models") or {}).items():
        users = ", ".join(model.get("used_by") or []) or "vision"
        if not model.get("present"):
            problems.append(f"The model file {filename} ({users} backend) is missing.")
        elif model.get("sha256_ok") is False:
            problems.append(f"The model file {filename} ({users} backend) is damaged (checksum).")
    if backends.get("active_error"):
        problems.append(str(backends["active_error"]))

    monitors = report.get("monitors") or {}
    if monitors.get("count") == 1 and not monitors.get("virtual"):
        problems.append(
            "Only one monitor detected: switching needs two or more "
            "(walk-away and privacy features still work)."
        )

    cameras = report.get("cameras") or {}
    if cameras.get("probed") and cameras.get("devices") == []:
        hint = " (it may be in use by Eye Tracker itself)" if _instance_running(report) else ""
        problems.append(f"No camera delivered a picture{hint}.")

    platform_info = report.get("platform") or {}
    permissions = platform_info.get("permissions") or {}
    if permissions.get("camera") is False:
        problems.append("Camera access is blocked by the operating system's privacy settings.")
    if platform_info.get("accessibility") == "stale":
        problems.append(
            "Accessibility permission was granted to an earlier version, so keyboard focus "
            "cannot follow your gaze: in System Settings > Privacy & Security > "
            "Accessibility remove Eye Tracker with '-' and add it again."
        )
    elif permissions.get("accessibility") is False:
        problems.append(
            "Accessibility permission is missing: keyboard focus cannot follow your gaze."
        )
    capabilities = platform_info.get("capabilities") or {}
    if capabilities and not capabilities.get("cursor", True):
        problems.append(
            "This session does not allow moving the cursor. On Wayland, sway and Hyprland "
            "work directly; elsewhere install ydotool 1.x and run ydotoold."
        )
    if capabilities and not capabilities.get("lock", True):
        problems.append("Locking the screen is not supported here.")

    hotkeys = report.get("hotkeys") or {}
    if hotkeys.get("enabled") and hotkeys.get("note"):
        problems.append(f"Global hotkeys: {hotkeys['note']}")
    for name, error in (hotkeys.get("invalid") or {}).items():
        problems.append(f"Hotkey {name} is invalid: {error}")
    if hotkeys.get("enabled"):
        for name, conflict in (hotkeys.get("layout_conflicts") or {}).items():
            problems.append(f"Hotkey {name} cannot be used: {conflict}.")
        registration = hotkeys.get("registration")
        for name, result in (registration if isinstance(registration, dict) else {}).items():
            reason = str(result).removeprefix("not registered: ")
            if reason != result and name not in (hotkeys.get("layout_conflicts") or {}):
                problems.append(f"Hotkey {name} could not be registered: {reason}.")

    autostart_info = report.get("autostart") or {}
    if autostart_info.get("status") == "stale":
        problems.append(
            "Start at login points to a copy of Eye Tracker that no longer exists; "
            "turn it off and on again."
        )

    settings = report.get("settings") or {}
    if settings.get("valid") is False:
        problems.append("The settings file is not valid JSON; defaults are used.")

    calibration = report.get("calibration") or {}
    # A calibration only serves monitor switching: none is missing with switching
    # turned off, or with one real monitor (reported above already).
    non_default = settings.get("non_default") or {}
    switching_wanted = non_default.get("switching.enabled", True) is not False
    one_monitor = monitors.get("count") == 1 and not monitors.get("virtual")
    if calibration.get("exists") is False:
        if switching_wanted and not one_monitor:
            # Spelled as this copy is run: release packages have no "eye-tracker" command.
            command = _redacted_command(cli_command("calibrate"))
            problems.append(
                f"Not calibrated yet: choose 'Calibrate...' in the tray menu or run '{command}'."
            )
    elif calibration.get("valid") is False:
        problems.append("The calibration file is damaged; please recalibrate.")
    elif calibration.get("compatible") is False:
        problems.append(f"The calibration no longer applies ({calibration.get('reason')}).")
    return problems


def _instance_running(report: dict[str, Any]) -> bool:
    return bool((report.get("app") or {}).get("instance_running"))


# ------------------------------------------------------------------ formatting
def _format_section(key: str, section: Any) -> list[str]:
    if not isinstance(section, dict):
        return [f"  {section}"]
    if key == "monitors" and "items" in section:
        rows = []
        for m in section["items"]:
            x, y, w, h = m["rect"]
            flags = " primary" if m.get("primary") else ""
            rows.append(
                (
                    f"#{m['index']}",
                    f"{m['name']}  {w}x{h} at ({x}, {y}){flags}, scale {m['scale']:g}, "
                    f"{m.get('logical_dpi', '?')} dpi",
                )
            )
        if "layout_signature" in section:
            rows.append(("layout", str(section["layout_signature"])))
        if section.get("virtual"):
            rows.append(("note", "virtual screen of a headless Qt platform, not a real monitor"))
        return _rows(rows)
    if key == "cameras" and section.get("probed"):
        devices = section.get("devices") or []
        base = {k: v for k, v in section.items() if k != "devices"}
        lines = _rows(_flatten(base))
        if not devices:
            lines.append("  (no camera found)")
        lines.extend(
            f"  camera {d['index']:<12} {d['name']} ({d['width']}x{d['height']})" for d in devices
        )
        return lines
    return _rows(_flatten(section))


def _flatten(section: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in section.items():
        label = f"{prefix}{key}"
        if isinstance(value, dict):
            if not value:
                rows.append((label, _EMPTY))
            else:
                rows.extend(_flatten(value, prefix=f"{label}."))
        else:
            rows.append((label, _format_value(value)))
    return rows


def _format_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(v) for v in value) if value else _EMPTY
    return str(value) or _EMPTY


def _rows(rows: Sequence[tuple[str, str]]) -> list[str]:
    if not rows:
        return [f"  {_EMPTY}"]
    width = min(max(len(k) for k, _ in rows), 34)
    return [f"  {k:<{width}}  {v}" for k, v in rows]


# ------------------------------------------------------------------ helpers
def _ensure_gui_app() -> Any:
    """The running ``QGuiApplication``, creating one if no Qt application exists.

    Returns ``None`` when only a ``QCoreApplication`` exists (no screens then).
    """
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QGuiApplication

    app = QCoreApplication.instance()
    if app is None:
        if (
            sys.platform.startswith("linux")
            and not os.environ.get("DISPLAY")
            and not os.environ.get("WAYLAND_DISPLAY")
            and not os.environ.get("QT_QPA_PLATFORM")
        ):
            # Without a display Qt would abort the process (e.g. over SSH);
            # the offscreen platform still lets everything else be reported.
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            _State.no_display = True
        services = _platform_services()
        services.prepare_process()
        app = QGuiApplication([sys.argv[0] if sys.argv and sys.argv[0] else "eye-tracker"])
        _State.owned_app.append(app)
        # macOS: no Dock icon for a command-line report.
        accessory = getattr(services, "set_accessory_app", None)
        if callable(accessory):
            accessory()
    return app if isinstance(app, QGuiApplication) else None


def _get_setting(settings: Settings, dotted: str) -> Any:
    section_name, _, field_name = dotted.partition(".")
    section = getattr(settings, section_name)
    if is_dataclass(section) and field_name in {f.name for f in fields(section)}:
        return getattr(section, field_name)
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact(value: str | os.PathLike[str] | None) -> str | None:
    """Replace the home directory with ``~`` wherever it appears in ``value``.

    Not only as a prefix: paths also turn up quoted (Explorer's "Copy as path"
    in *Pause while these apps run*), inside OS error messages and with either
    slash. A longer name that merely starts with the home path (``/home/al``
    in ``/home/alice``) is left alone.
    """
    if value is None:
        return None
    text = os.fspath(value)
    pattern = _home_pattern()
    return pattern.sub("~", text) if pattern is not None else text


def _home_pattern() -> re.Pattern[str] | None:
    """Regex matching every spelling of the home directory, longest first."""
    homes = {str(Path.home())}
    with contextlib.suppress(OSError, RuntimeError):
        homes.add(str(Path.home().resolve()))  # e.g. /home -> /var/home
    alternatives = []
    for home in sorted(homes, key=len, reverse=True):
        parts = re.split(r"[\\/]+", home)
        if len("".join(parts)) < 2:
            continue  # "/" or "C:\" alone would redact everything
        # Either slash matches either separator ("C:/Users/..." from Qt).
        alternatives.append(r"[\\/]+".join(re.escape(part) for part in parts))
    if not alternatives:
        return None
    # Case-insensitive file systems: C:\USERS\ALICE is the same directory.
    flags = re.IGNORECASE if os.name == "nt" or sys.platform == "darwin" else 0
    # Not followed by a character that would continue the last name (alice2, alice.old).
    return re.compile("(?:" + "|".join(alternatives) + r")(?![\w.\-])", flags)


def _redact_tree(value: Any) -> Any:
    """``value`` with :func:`_redact` applied to every string in nested dicts and lists."""
    pattern = _home_pattern()
    if pattern is None:
        return value

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            return pattern.sub("~", item)
        if isinstance(item, dict):
            return {key: walk(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [walk(child) for child in item]
        return item

    return walk(value)


def _describe_device(device: str) -> str:
    """A camera setting as a report may show it: no directories, no serial number."""
    spec = device.strip()
    if spec.isdigit():
        return f"camera {spec}"
    if not spec:
        return "none"
    name = Path(spec).name
    if spec.startswith("/dev/"):
        # /dev/videoN or a udev link; by-id names end in the camera's serial.
        if "/by-id/" in spec:
            name = _BY_ID_SERIAL.sub(r"_*\1", name)
        return f"device {name}"
    return f"file {name}"


# ======================================================================== bench
class BenchError(RuntimeError):
    """The benchmark could not run (no frames, camera unavailable, ...)."""


def run_bench(
    seconds: float = 5.0,
    device: str = "0",
    backend: str = "auto",
    *,
    width: int = 640,
    height: int = 480,
    api: str = "auto",
    idle_fps: float | None = None,
) -> dict[str, Any]:
    """Measure the capture + analysis loop without any UI.

    Two modes run for ``seconds`` each:

    * ``max``: every frame is analysed as fast as possible (raw throughput);
    * ``idle``: the balanced profile's idle rate with the motion gate on, which
      is how the app runs while you sit still in front of one monitor.

    ``device`` is a camera index or a video/image path. Reports per mode:
    fps, analysed fps, inference ms (p50/p95/mean), process CPU as a share of
    the whole machine and of one core, and how often a face was seen.

    Raises:
        BenchError: The source cannot be opened or delivers no frames.
        eye_tracker.vision.camera.CameraError: Invalid device.
        eye_tracker.vision.backends.BackendUnavailable: No usable backend.
    """
    if not seconds > 0:
        raise ValueError("seconds must be positive")
    from .engine.scheduler import PROFILES
    from .vision.backends import create_backend
    from .vision.camera import open_source
    from .vision.motion import MotionGate

    rate = float(idle_fps) if idle_fps else PROFILES["balanced"]["idle"]
    source = open_source(device, width, height, api, realtime=True)
    if not source.open():
        raise BenchError(source.last_error or f"Could not open {_describe_device(device)}")
    try:
        engine = create_backend(backend)
        try:
            stamp = _Timestamps()
            frame = None
            # The first inferences are slow (lazy initialisation); keep them out.
            for _ in range(3):
                frame = source.read()
                if frame is not None:
                    engine.process(frame, stamp.next())
            if frame is None:
                raise BenchError(source.last_error or "The video source delivered no frames")
            frame_size = [int(frame.shape[1]), int(frame.shape[0])]
            modes = {
                "max": _bench_mode(source, engine, seconds, None, None, stamp),
                "idle": _bench_mode(source, engine, seconds, rate, MotionGate(), stamp),
            }
        finally:
            engine.close()
    finally:
        source.release()
    return {
        "backend": engine.name,
        "feature_version": engine.feature_version,
        "device": _describe_device(device),
        "frame_size": frame_size,
        "cpu_count": os.cpu_count() or 1,
        "seconds_per_mode": seconds,
        "modes": modes,
    }


def format_bench(result: dict[str, Any]) -> str:
    """Render a :func:`run_bench` result as a small table."""
    modes = result.get("modes", {})
    columns = [(name, modes[name]) for name in ("max", "idle") if name in modes]

    def header(mode: dict[str, Any]) -> str:
        target = mode.get("target_fps")
        return "max speed" if target is None else f"idle ({target:g} fps)"

    def ms(stats: dict[str, Any] | None) -> str:
        if not stats or stats.get("p50") is None:
            return _EMPTY
        return f"{stats['p50']:.1f} / {stats['p95']:.1f} ms"

    def pct(value: float | None) -> str:
        return _EMPTY if value is None else f"{value:.1f} %"

    rows = [
        ("frames per second", [f"{m['fps']:.1f}" for _, m in columns]),
        ("analysed per second", [f"{m['analysed_fps']:.1f}" for _, m in columns]),
        ("inference p50 / p95", [ms(m.get("inference_ms")) for _, m in columns]),
        ("CPU, whole machine", [pct(m.get("cpu_percent")) for _, m in columns]),
        ("CPU, % of one core", [pct(m.get("cpu_percent_core")) for _, m in columns]),
        ("face seen", [pct(_percent(m.get("face_ratio"))) for _, m in columns]),
    ]
    w, h = result.get("frame_size", ["?", "?"])
    lines = [
        f"{APP_NAME} benchmark - {result.get('backend')} backend "
        f"({result.get('feature_version')}), {result.get('device')} at {w}x{h}, "
        f"{result.get('cpu_count')} logical CPUs",
        "",
        f"{'':<22}" + "".join(f"{header(m):<18}" for _, m in columns),
    ]
    lines.extend(f"{label:<22}" + "".join(f"{v:<18}" for v in values) for label, values in rows)
    return "\n".join(line.rstrip() for line in lines) + "\n"


class _Timestamps:
    """Strictly increasing timestamps for the backend (coarse clocks repeat values)."""

    def __init__(self) -> None:
        self._last = float("-inf")

    def next(self) -> float:
        self._last = max(time.monotonic(), self._last + 1e-3)
        return self._last


def _bench_mode(
    source: Any,
    engine: Any,
    seconds: float,
    target_fps: float | None,
    gate: Any,
    stamp: _Timestamps,
) -> dict[str, Any]:
    import numpy as np
    import psutil

    process = psutil.Process()
    period = 1.0 / target_fps if target_fps else 0.0
    inference_ms: list[float] = []
    read_ms: list[float] = []
    frames = analysed = faces = failures = 0
    have_observation = False

    cpu_start = process.cpu_times()
    start = time.perf_counter()
    next_slot = start
    while True:
        now = time.perf_counter()
        if now - start >= seconds:
            break
        if period:
            if now < next_slot:
                time.sleep(min(next_slot - now, 0.05))
                continue
            # A fixed schedule; after a stall, restart it rather than burst.
            next_slot = max(next_slot + period, now)
        t0 = time.perf_counter()
        frame = source.read()
        read_ms.append((time.perf_counter() - t0) * 1000.0)
        if frame is None:
            failures += 1
            if failures >= 5:
                raise BenchError(source.last_error or "The video source stopped delivering frames")
            continue
        failures = 0
        frames += 1
        if gate is not None and have_observation and not gate.should_process(frame, now):
            continue
        t0 = time.perf_counter()
        observation = engine.process(frame, stamp.next())
        inference_ms.append((time.perf_counter() - t0) * 1000.0)
        analysed += 1
        have_observation = True
        if observation.face_count > 0:
            faces += 1
        if gate is not None:
            gate.mark_processed(frame, now, observation.face_box)
    elapsed = max(time.perf_counter() - start, 1e-9)
    cpu_end = process.cpu_times()
    cpu_seconds = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cores = os.cpu_count() or 1

    def stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"p50": None, "p95": None, "mean": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "mean": round(float(arr.mean()), 3),
        }

    return {
        "target_fps": target_fps,
        "seconds": round(elapsed, 3),
        "frames": frames,
        "analysed": analysed,
        "fps": round(frames / elapsed, 2),
        "analysed_fps": round(analysed / elapsed, 2),
        "skip_ratio": round(1.0 - analysed / frames, 3) if frames else None,
        "inference_ms": stats(inference_ms),
        "read_ms": stats(read_ms),
        "cpu_percent": round(100.0 * cpu_seconds / elapsed / cores, 2),
        "cpu_percent_core": round(100.0 * cpu_seconds / elapsed, 2),
        "face_ratio": round(faces / analysed, 3) if analysed else None,
    }


def _percent(ratio: float | None) -> float | None:
    return None if ratio is None else 100.0 * ratio
