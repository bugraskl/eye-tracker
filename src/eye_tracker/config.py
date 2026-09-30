"""User settings: typed dataclasses persisted as JSON.

Loading is forgiving by design. Unknown keys are ignored, missing keys fall back
to defaults, wrongly typed values are replaced by the default and out-of-range
numbers are clamped. A hand-edited or older config file can therefore never
prevent the app from starting; problems are reported through the log.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CONFIG_VERSION = 1


def _opt(
    default: Any,
    *,
    lo: float | None = None,
    hi: float | None = None,
    choices: tuple[str, ...] | None = None,
    doc: str = "",
) -> Any:
    return field(default=default, metadata={"lo": lo, "hi": hi, "choices": choices, "doc": doc})


@dataclass
class GeneralSettings:
    backend: str = _opt(
        "auto",
        choices=("auto", "facemesh", "lite"),
        doc="Vision backend. 'facemesh' tracks 478 face landmarks including the irises "
        "(head pose + eye direction); 'lite' uses 5 landmarks (head pose only, lowest CPU). "
        "'auto' picks facemesh.",
    )
    start_paused: bool = _opt(False, doc="Start with tracking paused.")
    first_run_done: bool = _opt(False, doc="Set after the first-run wizard completes.")
    notifications: bool = _opt(True, doc="Show tray notifications.")
    log_level: str = _opt("INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))


@dataclass
class CameraSettings:
    device: str = _opt(
        "0",
        doc="Camera index ('0', '1', …) or a video/image file path (useful for testing).",
    )
    width: int = _opt(640, lo=160, hi=1920, doc="Requested capture width.")
    height: int = _opt(480, lo=120, hi=1080, doc="Requested capture height.")
    api: str = _opt(
        "auto",
        choices=("auto", "dshow", "msmf", "avfoundation", "v4l2", "any"),
        doc="OpenCV capture API.",
    )


@dataclass
class PerformanceSettings:
    profile: str = _opt(
        "balanced",
        choices=("eco", "balanced", "responsive"),
        doc="Frame-rate profile. Eco minimises CPU, responsive minimises latency.",
    )
    motion_gate: bool = _opt(True, doc="Skip face analysis when the camera image has not changed.")
    motion_threshold: float = _opt(
        2.0, lo=0.1, hi=20.0, doc="Mean grey-level change that counts as motion."
    )


@dataclass
class SwitchingSettings:
    enabled: bool = _opt(True, doc="Move the cursor to the monitor you look at.")
    dwell_ms: int = _opt(
        300, lo=0, hi=3000, doc="How long you must look at another monitor before switching."
    )
    hysteresis: float = _opt(
        0.06,
        lo=0.0,
        hi=0.5,
        doc="How far (fraction of the monitor size) the gaze must cross into another monitor.",
    )
    off_screen_margin: float = _opt(
        0.35,
        lo=0.05,
        hi=2.0,
        doc="Gaze further than this (fraction of the monitor diagonal) outside every "
        "monitor is treated as looking away (phone, desk) and ignored.",
    )
    cooldown_ms: int = _opt(600, lo=0, hi=5000, doc="Minimum time between two switches.")
    mouse_grace_ms: int = _opt(
        1500, lo=0, hi=10000, doc="No switching for this long after you move the mouse."
    )
    typing_grace_ms: int = _opt(
        2000, lo=0, hi=10000, doc="No switching for this long after you type."
    )
    reading_grace_ms: int = _opt(
        6000,
        lo=0,
        hi=60000,
        doc="After you typed while looking at another monitor (e.g. copying from a document "
        "there), switching to that monitor waits this long after your last keystroke instead "
        "of the typing grace, so reading pauses do not move your keyboard focus. 0 turns this "
        "off.",
    )
    cursor_target: str = _opt(
        "last",
        choices=("last", "center", "gaze"),
        doc="Where the cursor lands: last position on that monitor, its centre, or the "
        "estimated gaze point.",
    )
    focus_window: bool = _opt(
        True, doc="Also give keyboard focus to the last-used window on the target monitor."
    )
    smoothing: float = _opt(
        0.5, lo=0.0, hi=1.0, doc="Gaze smoothing strength (0 = raw, 1 = heavy)."
    )


@dataclass
class PresenceSettings:
    enabled: bool = _opt(True, doc="React when you walk away from the computer.")
    action: str = _opt(
        "lock",
        choices=("none", "notify", "display_off", "lock", "lock_and_display_off"),
        doc="What to do when you are away.",
    )
    away_timeout_s: int = _opt(
        45, lo=5, hi=3600, doc="Seconds without a face (and without input) before acting."
    )
    warning_s: int = _opt(
        10, lo=0, hi=120, doc="Countdown shown before the action; any input cancels it."
    )
    require_input_idle: bool = _opt(
        True, doc="Keyboard or mouse activity counts as presence even without a face."
    )
    wake_on_return: bool = _opt(
        True, doc="Turn the displays back on when you return (if they were only switched off)."
    )


@dataclass
class PrivacySettings:
    pause_when_locked: bool = _opt(True, doc="Release the camera while the session is locked.")
    yield_camera: bool = _opt(
        True,
        doc="Release the camera while another app is using it (Windows and Linux only).",
    )
    pause_for_apps: list[str] = field(
        default_factory=list,
        metadata={"doc": "Process names that pause tracking while running (e.g. 'obs64.exe')."},
    )
    shoulder_guard: bool = _opt(False, doc="React when a second face appears behind you.")
    guard_action: str = _opt(
        "curtain", choices=("notify", "curtain", "lock"), doc="Shoulder guard reaction."
    )
    guard_delay_s: float = _opt(
        2.0, lo=0.5, hi=30.0, doc="Seconds a second face must be visible before reacting."
    )


#: Modifiers of the default hotkeys by ``sys.platform``; the keys are T (pause /
#: resume tracking), P (privacy mode) and C (calibrate) everywhere. Each set was
#: chosen so that the combinations type no character and are no stock OS shortcut:
#:
#: * Windows reports AltGr as Ctrl+Alt, and the layout tables of the stock Windows
#:   layouts show Ctrl+Alt(+Shift)+T, P and C typing characters on 19-47 layouts each
#:   (Ctrl+Alt+T is '₺' on Turkish Q, Ctrl+Alt+C 'ć' on Polish (Programmers),
#:   Ctrl+Alt+P 'ö' on US-International). No Windows layout uses the Win key as a
#:   character modifier, and the shell's own Win shortcuts (Game Bar Win+Alt+*,
#:   Win+Ctrl+*, the Office key Ctrl+Alt+Shift+Win) leave Ctrl+Alt+Win+T/P/C free.
#: * Linux desktops open a terminal on Ctrl+Alt+T and switch virtual terminals on
#:   Ctrl+Alt+F1-F12. X11 keeps AltGr a modifier of its own, so Ctrl+Alt+Shift
#:   never types, and no major desktop binds it to T, P or C.
#: * macOS has no AltGr, Control+Option+letter types nothing, and macOS binds no
#:   ⌃⌥+letter shortcut (only VoiceOver, while it runs, uses ⌃⌥ as its prefix).
_HOTKEY_MODIFIERS: dict[str, str] = {"win32": "ctrl+alt+meta", "darwin": "ctrl+alt"}
_HOTKEY_MODIFIERS_OTHER = "ctrl+alt+shift"  # Linux/X11 and other Unix desktops


def _default_hotkey(key: str) -> str:
    """The default hotkey for ``key`` on the running platform (see above)."""
    return f"{_HOTKEY_MODIFIERS.get(sys.platform, _HOTKEY_MODIFIERS_OTHER)}+{key}"


@dataclass
class HotkeySettings:
    enabled: bool = _opt(True, doc="Register global hotkeys.")
    # Factories (not plain defaults) so the platform is looked up when settings are
    # created, which also lets tests check every platform's defaults.
    toggle_tracking: str = field(
        default_factory=lambda: _default_hotkey("t"),
        metadata={"doc": "Pause / resume tracking."},
    )
    toggle_privacy: str = field(
        default_factory=lambda: _default_hotkey("p"),
        metadata={"doc": "Privacy mode (camera fully off)."},
    )
    recalibrate: str = field(
        default_factory=lambda: _default_hotkey("c"),
        metadata={"doc": "Start calibration."},
    )


@dataclass
class LearningSettings:
    adaptive: bool = _opt(True, doc="Refine the calibration from your natural mouse use.")
    max_samples: int = _opt(400, lo=0, hi=5000, doc="Maximum number of learned samples kept.")
    drift_alerts: bool = _opt(True, doc="Suggest recalibration when accuracy drops.")


@dataclass
class UISettings:
    show_gaze_overlay: bool = _opt(False, doc="Draw a dot where you are looking (testing aid).")


@dataclass
class Settings:
    version: int = CONFIG_VERSION
    general: GeneralSettings = field(default_factory=GeneralSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    switching: SwitchingSettings = field(default_factory=SwitchingSettings)
    presence: PresenceSettings = field(default_factory=PresenceSettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    hotkeys: HotkeySettings = field(default_factory=HotkeySettings)
    learning: LearningSettings = field(default_factory=LearningSettings)
    ui: UISettings = field(default_factory=UISettings)

    # ------------------------------------------------------------------ helpers
    def copy(self) -> Settings:
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        settings = cls()
        if not isinstance(data, dict):
            log.warning("Settings root is not an object; using defaults")
            return settings
        _apply(settings, data, prefix="")
        settings.version = CONFIG_VERSION
        return settings

    @classmethod
    def load(cls, path: Path) -> Settings:
        """Load settings, falling back to defaults on any error (never raises)."""
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls()
        except OSError as exc:
            log.warning("Could not read %s: %s; using defaults", path, exc)
            return cls()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            backup = path.with_suffix(path.suffix + ".corrupt")
            log.warning("Invalid JSON in %s (%s); moved to %s", path, exc, backup.name)
            with contextlib.suppress(OSError):
                os.replace(path, backup)
            return cls()
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        """Atomically write the settings as pretty-printed JSON."""
        atomic_write_text(path, json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def describe_settings() -> list[dict[str, Any]]:
    """Flat description of every setting (used by docs and the settings UI)."""
    rows: list[dict[str, Any]] = []
    root = Settings()
    for sec in fields(root):
        section = getattr(root, sec.name)
        if not is_dataclass(section):
            continue
        for f in fields(section):
            meta = f.metadata
            rows.append(
                {
                    "key": f"{sec.name}.{f.name}",
                    "default": getattr(section, f.name),
                    "lo": meta.get("lo"),
                    "hi": meta.get("hi"),
                    "choices": meta.get("choices"),
                    "doc": meta.get("doc", ""),
                }
            )
    return rows


# ---------------------------------------------------------------------- internals
def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


def _default_of(f: Any) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return None


def _apply(target: Any, data: dict[str, Any], prefix: str) -> None:
    for f in fields(target):
        if f.name == "version" or f.name not in data:
            continue
        key = f"{prefix}{f.name}"
        value = data[f.name]
        current = getattr(target, f.name)
        if is_dataclass(current):
            if isinstance(value, dict):
                _apply(current, value, prefix=f"{key}.")
            else:
                log.warning("Setting %s should be an object; ignored", key)
            continue
        coerced = _coerce(key, value, _default_of(f), f.metadata)
        setattr(target, f.name, coerced)


def _coerce(key: str, value: Any, default: Any, meta: Any) -> Any:
    lo, hi, choices = meta.get("lo"), meta.get("hi"), meta.get("choices")
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        log.warning("Setting %s expects true/false; using default %r", key, default)
        return default
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            log.warning("Setting %s expects a number; using default %r", key, default)
            return default
        num: float = float(value)
        if math.isnan(num) or math.isinf(num):
            log.warning("Setting %s must be finite; using default %r", key, default)
            return default
        if lo is not None and num < lo:
            log.warning("Setting %s=%r below minimum %r; clamped", key, value, lo)
            num = float(lo)
        if hi is not None and num > hi:
            log.warning("Setting %s=%r above maximum %r; clamped", key, value, hi)
            num = float(hi)
        return round(num) if isinstance(default, int) else num
    if isinstance(default, str):
        if not isinstance(value, str):
            log.warning("Setting %s expects a string; using default %r", key, default)
            return default
        if choices and value not in choices:
            log.warning("Setting %s=%r not one of %s; using default", key, value, choices)
            return default
        return value
    if isinstance(default, list):
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return [v for v in value if v.strip()]
        log.warning("Setting %s expects a list of strings; using default", key)
        return list(default)
    return value
