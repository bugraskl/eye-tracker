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

#: Format version written into the settings file. Files of every earlier version
#: keep loading; :func:`_migrate` updates them:
#:
#: 1. The first builds.
#: 2. Today's default hotkeys (see ``_HOTKEY_MODIFIERS``). A version 1 file whose
#:    three hotkeys are still an earlier build's untouched defaults gets today's
#:    defaults. The next save writes version 2, so this happens once: hotkeys
#:    chosen afterwards are kept even if they equal an old default.
CONFIG_VERSION = 2

#: Vision backend names of earlier versions and what they are called now. The
#: MediaPipe-runtime backend became the self-contained "facemesh" one (same
#: landmark model), the OpenCV/YuNet one became "lite".
LEGACY_BACKENDS: dict[str, str] = {"mediapipe": "facemesh", "opencv": "lite"}
#: The hotkey defaults of the first builds (``hotkeys.toggle_tracking``,
#: ``toggle_privacy``, ``recalibrate``) on every platform. Ctrl+Alt+letter is
#: AltGr+letter on many Windows layouts, opens a terminal on Linux desktops and
#: is a Rectangle / Magnet shortcut on macOS.
LEGACY_HOTKEYS: tuple[str, str, str] = ("ctrl+alt+t", "ctrl+alt+p", "ctrl+alt+c")
#: The Linux (and other X11 desktop) defaults of later version 1 builds. Alt+Shift
#: and Ctrl+Shift are common keyboard-layout switches there, which make this chord
#: impossible to press, and JetBrains IDEs use it.
LEGACY_HOTKEYS_X11: tuple[str, str, str] = (
    "ctrl+alt+shift+t",
    "ctrl+alt+shift+p",
    "ctrl+alt+shift+c",
)
#: The first file version that may hold old default hotkeys on purpose.
_HOTKEY_DEFAULTS_VERSION = 2
_HOTKEY_FIELDS = ("toggle_tracking", "toggle_privacy", "recalibrate")


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
    first_run_done: bool = _opt(
        False,
        doc="Set after the first-run wizard completes. Until then a walk-away action that "
        "locks or turns the displays off only shows a notification.",
    )
    notifications: bool = _opt(True, doc="Show tray notifications.")
    log_level: str = _opt(
        "INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        doc="Log file verbosity. DEBUG helps with bug reports; logs never contain images or keys.",
    )


@dataclass
class CameraSettings:
    device: str = _opt(
        "0",
        doc="Camera index ('0', '1', …) or a local video/image file path (useful for "
        "testing). On Linux also /dev/videoN or a stable link such as /dev/v4l/by-id/…, "
        "which keeps finding a camera that was re-plugged under another number. URLs and "
        "other network sources are refused: Eye Tracker works fully offline.",
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
class PaneSettings:
    enabled: bool = _opt(
        False,
        doc="Experimental: also move keyboard focus between split panes of supported "
        "terminals (tmux, WezTerm, Windows Terminal) on the monitor you are already on. "
        "Only panes large enough for your calibration's accuracy take part.",
    )
    dwell_ms: int = _opt(
        600, lo=100, hi=5000, doc="How long you must look at another pane before it gets focus."
    )
    typing_grace_ms: int = _opt(
        3000, lo=0, hi=20000, doc="No pane switching for this long after you type."
    )
    reading_grace_ms: int = _opt(
        8000,
        lo=0,
        hi=60000,
        doc="After you typed while looking at another pane (e.g. reading its output), "
        "switching to that pane waits this long after your last keystroke instead of the "
        "typing grace. 0 turns this off.",
    )
    cooldown_ms: int = _opt(1000, lo=0, hi=10000, doc="Minimum time between two pane switches.")
    after_monitor_switch_ms: int = _opt(
        1500,
        lo=0,
        hi=10000,
        doc="No pane switching for this long after the cursor moved to another monitor.",
    )
    precision: float = _opt(
        2.5,
        lo=1.0,
        hi=6.0,
        doc="A pane takes part only if it is at least this many times your gaze error "
        "(measured by the calibration) wide, for panes side by side, or tall, for stacked "
        "panes. Higher is safer, lower allows smaller panes.",
    )
    hysteresis: float = _opt(
        0.5,
        lo=0.0,
        hi=3.0,
        doc="How far past the divider the gaze must be, as a fraction of your gaze error.",
    )
    min_pane_px: int = _opt(
        240, lo=0, hi=4000, doc="Panes narrower (or lower) than this many pixels never take part."
    )
    tmux: bool = _opt(True, doc="Follow tmux panes (through the tmux command, also in WSL).")
    wezterm: bool = _opt(True, doc="Follow WezTerm panes (through the wezterm command).")
    windows_terminal: bool = _opt(
        True, doc="Follow Windows Terminal panes (Windows only, through UI Automation)."
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
        45,
        lo=5,
        hi=3600,
        doc="Seconds without a face (and without input) before acting; the countdown runs "
        "during the last seconds of this time.",
    )
    warning_s: int = _opt(
        10,
        lo=0,
        hi=120,
        doc="Countdown shown before the action (0 = act without a countdown). Being seen "
        "by the camera cancels it, and so does keyboard or mouse use while "
        "require_input_idle is on.",
    )
    require_input_idle: bool = _opt(
        True,
        doc="Keyboard or mouse activity counts as presence even without a face (not on "
        "Wayland outside GNOME, where the system does not report it).",
    )
    wake_on_return: bool = _opt(
        True, doc="Turn the displays back on when you return (if they were only switched off)."
    )


@dataclass
class PrivacySettings:
    remember_privacy_mode: bool = _opt(
        True,
        doc="Privacy mode stays on when Eye Tracker or the computer restarts (also after an "
        "update), so the camera never comes back on by itself.",
    )
    pause_when_locked: bool = _opt(True, doc="Release the camera while the session is locked.")
    yield_camera: bool = _opt(
        True,
        doc="Release the camera while another app is using it (Windows and Linux only). "
        "Walk-away detection and the shoulder guard pause meanwhile.",
    )
    pause_for_apps: list[str] = field(
        default_factory=list,
        metadata={
            "doc": "Process names that pause tracking while running (e.g. 'obs64.exe'). The "
            "camera is released, so walk-away detection and the shoulder guard pause too."
        },
    )
    shoulder_guard: bool = _opt(False, doc="React when a second face appears behind you.")
    guard_action: str = _opt(
        "curtain",
        choices=("notify", "curtain", "lock"),
        doc="Shoulder guard reaction. After you unlock a lock it caused, 'lock' covers the "
        "screens instead of locking again until nobody has looked over your shoulder for "
        "5 minutes.",
    )
    guard_delay_s: float = _opt(
        2.0, lo=0.5, hi=30.0, doc="Seconds a second face must be visible before reacting."
    )


#: Modifiers of the default hotkeys by ``sys.platform``; the keys are T (pause /
#: resume tracking), P (privacy mode) and C (calibrate) everywhere. A global hotkey
#: takes its combination away from every other application, so the defaults must
#: type no character, stay pressable while a keyboard-layout switch is configured,
#: and be no default shortcut of the system or of the tools this app's users are
#: likely to run. Ctrl+Alt+Meta (Ctrl+Alt+Win, ⌃⌥⌘, Ctrl+Alt+Super) passes on
#: every platform; what was checked:
#:
#: * Windows reports AltGr as Ctrl+Alt, and the layout tables of the stock Windows
#:   layouts show Ctrl+Alt(+Shift)+T, P and C typing characters on 19-47 layouts each
#:   (Ctrl+Alt+T is '₺' on Turkish Q, Ctrl+Alt+C 'ć' on Polish (Programmers),
#:   Ctrl+Alt+P 'ö' on US-International). No Windows layout uses the Win key as a
#:   character modifier, and the shell's own Win shortcuts (Game Bar Win+Alt+*,
#:   Win+Ctrl+*, the Office key Ctrl+Alt+Shift+Win) leave Ctrl+Alt+Win+T/P/C free.
#: * Linux (X11): AltGr is a modifier of its own (Mod5), so Ctrl+Alt chords never
#:   type. GNOME, KDE Plasma, Xfce, Cinnamon and MATE open a terminal on Ctrl+Alt+T,
#:   switch virtual terminals on Ctrl+Alt+F1-F12 and bind Super+letter (Super+P:
#:   displays), but none binds Ctrl+Alt+Super+T, P or C, and X grabs match the
#:   modifiers exactly, so Super+P is not Ctrl+Alt+Super+P. Ctrl+Alt+Shift, the
#:   default of earlier builds, failed twice. With the XKB layout switches that
#:   bilingual users pick most, Alt+Shift (grp:alt_shift_toggle) and Ctrl+Shift
#:   (grp:ctrl_shift_toggle), whichever of the pair is pressed second switches the
#:   layout instead of adding its modifier (xkeyboard-config symbols/group), so the
#:   chord cannot be formed and arrives as Ctrl+Alt+T (a terminal) or Ctrl+Shift+T.
#:   And IDEs use it: the JetBrains keymap for Refactor This, Copy Reference and
#:   Introduce Functional Parameter (Ctrl+Alt+Shift+T/C/P), VS Code for Copy Path
#:   (Ctrl+Alt+Shift+C). Neither binds Ctrl+Alt+Super. The other usual switches
#:   (Super+Space, Caps Lock, Shift+Caps Lock) leave the defaults alone; the rare
#:   options that do (grp:ctrl_alt_toggle, the Win-key switches and selectors,
#:   altwin:ctrl_win, ...) are detected by the hotkey manager, which then reports
#:   the hotkey as unavailable instead of grabbing a chord that cannot be pressed.
#:   One limit remains where a desktop binds Super on its own (Xubuntu's menu):
#:   its grab takes the keyboard while Super is held, so the chord works only
#:   with Ctrl or Alt pressed first. The hotkey manager detects that and says so.
#: * macOS: Control+Option+Command+letter types nothing. Rectangle and Magnet, the
#:   window managers multi-monitor Mac users run most, take ⌃⌥ with arrows and
#:   letters by default (⌃⌥T Last Two Thirds, ⌃⌥C Center) and ⌃⌥⌘ only with ←/→
#:   (previous / next display); Rectangle's Spectacle scheme uses ⌥⌘ (⌥⌘C Center).
#:   Moom and BetterTouchTool use only the shortcuts their user records. macOS binds
#:   no ⌃⌥⌘+letter (VoiceOver, while it runs, reads every ⌃⌥ chord as a command of
#:   its own). Hotkeys are registered exclusively, so a combination that another
#:   app already holds is reported ("Hotkey unavailable: ⌃⌥⌘T is already in use by
#:   another application") instead of firing in both apps.
#:
#: The table stays per platform so that one platform's defaults can change without
#: touching the others; :data:`LEGACY_HOTKEYS` and :data:`LEGACY_HOTKEYS_X11` list
#: what earlier builds used, and :func:`_migrate` moves untouched old files over.
_HOTKEY_MODIFIERS: dict[str, str] = {"win32": "ctrl+alt+meta", "darwin": "ctrl+alt+meta"}
_HOTKEY_MODIFIERS_OTHER = "ctrl+alt+meta"  # Linux/X11 and other Unix desktops


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
    panes: PaneSettings = field(default_factory=PaneSettings)
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
        _apply(settings, _migrate(data), prefix="")
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
def _migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite values written by earlier versions (before validation).

    Returns ``data`` itself when nothing changes, otherwise a shallow copy with
    fresh copies of the sections that changed: the caller's dict is never
    modified.
    """
    general = data.get("general")
    backend = general.get("backend") if isinstance(general, dict) else None
    if isinstance(general, dict) and isinstance(backend, str) and backend in LEGACY_BACKENDS:
        renamed = LEGACY_BACKENDS[backend]
        log.info("Setting general.backend=%r is now called %r", backend, renamed)
        data = {**data, "general": {**general, "backend": renamed}}

    hotkeys = data.get("hotkeys")
    if isinstance(hotkeys, dict) and _file_version(data) < _HOTKEY_DEFAULTS_VERSION:
        stored = tuple(hotkeys.get(name) for name in _HOTKEY_FIELDS)
        if all(isinstance(v, str) for v in stored) and (
            tuple(str(v).strip().lower() for v in stored) in _earlier_default_hotkeys()
        ):
            # Written before today's defaults and all three still at an earlier
            # build's defaults: the user never chose them. A newer file may hold the
            # same values on purpose, which is why only old files are looked at.
            defaults = HotkeySettings()
            fresh = {name: getattr(defaults, name) for name in _HOTKEY_FIELDS}
            log.info(
                "Hotkeys at the defaults of an earlier version; using today's defaults (%s)",
                ", ".join(fresh.values()),
            )
            data = {**data, "hotkeys": {**hotkeys, **fresh}}
    return data


def _file_version(data: dict[str, Any]) -> int:
    """The format version a settings dict was written with.

    Files have always carried one; a missing or malformed value (a hand-written
    file) counts as the oldest format.
    """
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        return 1
    return version


def _earlier_default_hotkeys() -> tuple[tuple[str, str, str], ...]:
    """The hotkey trios that version 1 builds wrote as defaults on this platform."""
    if sys.platform in ("win32", "darwin"):
        return (LEGACY_HOTKEYS,)
    return (LEGACY_HOTKEYS, LEGACY_HOTKEYS_X11)


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
