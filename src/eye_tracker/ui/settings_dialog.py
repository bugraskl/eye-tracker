"""Settings dialog: every option of :class:`~eye_tracker.config.Settings`, grouped by task.

The dialog edits a *copy* of the controller's settings. Nothing changes until
the user presses OK or Apply, which hands the copy to
``Controller.apply_settings``. Each widget is bound to one settings key through a
small binding object that remembers what was loaded, so

* a value the user did not touch is written back exactly as it was loaded (no
  float drift from spin-box rounding), and
* settings changed elsewhere while the dialog is open (tray menu, IPC) refresh
  the untouched widgets without discarding the user's pending edits.

Tooltips come from the field documentation in :func:`eye_tracker.config.describe_settings`.
"""

from __future__ import annotations

import contextlib
import html
import logging
import math
import sys
import threading
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QDesktopServices,
    QFocusEvent,
    QFontDatabase,
    QGuiApplication,
    QHideEvent,
    QKeyEvent,
    QKeySequence,
    QPalette,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, paths
from ..cli import cli_command_text
from ..config import Settings, describe_settings
from ..platform import autostart
from ..platform.base import PlatformServices
from ..platform.hotkeys import Hotkey, HotkeyManager, format_hotkey, parse_hotkey
from ..types import TrackingState
from . import util
from .icons import app_icon
from .util import ACCENT, DANGER, WARNING, ui_scale

log = logging.getLogger(__name__)

__all__ = [
    "BACKEND_LABELS",
    "HOTKEY_ACTIONS",
    "CameraProbe",
    "HotkeyEdit",
    "SettingsDialog",
    "camera_in_use",
    "format_stats",
    "hotkey_text_from_event",
    "load_diagnostics_text",
    "modifier_preview",
    "platform_from_controller",
    "probe_cameras",
    "settings_from_controller",
]


#: Hotkey actions shown on the Hotkeys page: (settings field, label).
HOTKEY_ACTIONS: tuple[tuple[str, str], ...] = (
    ("toggle_tracking", "Pause / resume tracking"),
    ("toggle_privacy", "Privacy mode (camera off)"),
    ("recalibrate", "Recalibrate"),
)

#: Explanations under "Start at login" by ``autostart.status()`` value.
_AUTOSTART_NOTES: dict[str, str] = {
    "stale": (
        f"The login item starts a copy of {APP_NAME} that no longer exists or will be gone "
        "after a restart. Tick the box and apply to point it at this copy."
    ),
    "other-profile": (
        f"The login item starts {APP_NAME} with another settings folder (--config-dir). "
        "Tick the box and apply to start this one instead."
    ),
}

#: Vision backends offered in the settings (``general.backend`` values), best first.
BACKEND_LABELS: dict[str, str] = {
    "auto": "Automatic (recommended)",
    "facemesh": "Face mesh — head pose + eye direction (most accurate)",
    "lite": "Lite — head pose from face geometry only (lightest)",
}

# Settings metadata (range, choices, documentation) keyed by "section.field".
_META: dict[str, dict[str, Any]] = {row["key"]: row for row in describe_settings()}


# ============================================================ controller access
def settings_from_controller(controller: object) -> Settings:
    """A private copy of the controller's current settings.

    Uses ``controller.settings`` (attribute, property or method). Falls back to the
    settings file on disk, which the controller keeps up to date, rather than to
    defaults: applying defaults would silently reset the user's configuration.
    """
    value = getattr(controller, "settings", None)
    if callable(value):
        try:
            value = value()
        except Exception:
            log.debug("controller.settings() failed", exc_info=True)
            value = None
    if isinstance(value, Settings):
        return value.copy()
    log.warning("Controller exposes no settings; reading %s", paths.settings_file())
    return Settings.load(paths.settings_file())


def platform_from_controller(controller: object) -> PlatformServices:
    """The controller's platform services, or the process-wide singleton."""
    value = getattr(controller, "platform", None)
    if isinstance(value, PlatformServices):
        return value
    from ..platform import get_platform

    return get_platform()


def _capabilities(platform: PlatformServices) -> dict[str, bool]:
    try:
        return dict(platform.capabilities())
    except Exception:
        log.debug("capabilities() failed", exc_info=True)
        return {}


def format_stats(stats: object) -> str:
    """One line summarising ``Controller.stats_changed`` data.

    Example: ``"Tracking · 4.0 fps (target 4) · 9.5 ms/frame · 38 % skipped · CPU 0.6 %"``.
    Unknown or missing keys are left out, so this never raises.
    """
    if not isinstance(stats, dict):
        return ""
    parts: list[str] = []
    state = stats.get("state")
    label = stats.get("state_label")
    if isinstance(label, str) and label:
        parts.append(label)
    elif isinstance(state, TrackingState):
        parts.append(state.label)
    elif isinstance(state, str) and state:
        try:
            parts.append(TrackingState(state).label)
        except ValueError:
            parts.append(state)

    def number(key: str) -> float | None:
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) else None

    fps, target = number("fps"), number("target_fps")
    if fps is not None:
        text = f"{fps:.1f} fps"
        if target:
            text += f" (target {target:g})"
        parts.append(text)
    inference = number("inference_ms")
    if inference:
        parts.append(f"{inference:.1f} ms/frame")
    skip = number("skip_ratio")
    if skip is not None:
        parts.append(f"{skip * 100:.0f} % skipped")
    cpu = number("cpu_percent")
    if cpu is not None:
        parts.append(f"CPU {cpu:.1f} %")
    backend = stats.get("backend")
    if isinstance(backend, str) and backend:
        parts.append(backend)
    return " · ".join(parts)


def probe_cameras(max_index: int = 4, api: str = "auto", skip: Collection[int] = ()) -> list[Any]:
    """List cameras that deliver frames (runs on a worker thread; tests patch this).

    Indices in ``skip`` are never opened: the camera the tracker is using must
    not be probed, since closing the probe would tear down the tracker's capture
    of the same device on DirectShow.
    """
    from ..vision.camera import list_cameras

    return list(list_cameras(max_index, api, skip=skip))


def camera_in_use(controller: object) -> set[int]:
    """The camera index the controller has applied, as a ``skip`` set for probing.

    Uses the *applied* settings, not a value being edited in a dialog: that is
    the device the vision worker may hold open. On Linux a ``/dev/videoN`` path
    or a stable ``/dev/v4l/by-id/…`` link counts as the index it leads to.
    Empty for a video file.
    """
    device = str(settings_from_controller(controller).camera.device).strip()
    if device.isdigit():
        return {int(device)}
    if not device.startswith("/dev/"):
        return set()
    # Imported only for device paths: the camera module loads OpenCV.
    from ..vision.camera import device_index

    index = device_index(device)
    return {index} if index is not None else set()


def load_diagnostics_text(controller: object | None = None) -> str:
    """The ``doctor`` report as text (tests patch this).

    Imported lazily: the report module is only needed when the page is opened.
    Without a camera probe, so no camera light flashes. The controller's hotkey
    manager, when there is one, tells the report which hotkeys the OS accepted.
    """
    try:
        from .. import diagnostics

        collect = diagnostics.collect_report
        render = diagnostics.format_report
    except (ImportError, AttributeError) as exc:
        log.warning("Diagnostics module unavailable: %s", exc)
        return (
            "The diagnostics report is not available in this build.\n"
            f"Run '{cli_command_text('doctor')}' in a terminal instead."
        )
    manager = getattr(controller, "hotkey_manager", None)
    return str(
        render(
            collect(False, hotkey_manager=manager if isinstance(manager, HotkeyManager) else None)
        )
    )


# =================================================================== camera probe
class CameraProbe(QObject):
    """Runs :func:`probe_cameras` on a background thread.

    Opening camera devices can take seconds per index on Windows, which must not
    freeze the UI. ``finished`` delivers ``list[CameraInfo]`` on the main thread.
    """

    finished = Signal(object)
    _done = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: threading.Thread | None = None
        self._done.connect(self._on_done, Qt.ConnectionType.QueuedConnection)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, api: str = "auto", max_index: int = 4, skip: Collection[int] = ()) -> bool:
        """Start probing (never opening the indices in ``skip``).

        Returns ``False`` if a probe is already running.
        """
        if self.running:
            return False
        self._thread = threading.Thread(
            target=self._run,
            args=(api, max_index, frozenset(skip)),
            name="eye-tracker-camera-probe",
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, api: str, max_index: int, skip: frozenset[int]) -> None:
        try:
            cameras = probe_cameras(max_index, api, skip)
        except Exception:
            log.warning("Camera detection failed", exc_info=True)
            cameras = []
        try:
            self._done.emit(cameras)
        except RuntimeError:
            # The dialog (and this object) was closed while probing.
            log.debug("Camera probe finished after its owner was deleted")

    def _on_done(self, cameras: object) -> None:
        self._thread = None
        self.finished.emit(cameras)


# =================================================================== hotkey editor
_MODIFIER_KEYS = frozenset(
    k.value
    for k in (
        Qt.Key.Key_Control,
        Qt.Key.Key_Shift,
        Qt.Key.Key_Alt,
        Qt.Key.Key_AltGr,
        Qt.Key.Key_Meta,
        Qt.Key.Key_Super_L,
        Qt.Key.Key_Super_R,
        Qt.Key.Key_Hyper_L,
        Qt.Key.Key_Hyper_R,
    )
)


def _build_key_table() -> dict[int, str]:
    K = Qt.Key
    table: dict[int, str] = {
        K.Key_Space.value: "space",
        K.Key_Return.value: "enter",
        K.Key_Enter.value: "enter",
        K.Key_Tab.value: "tab",
        K.Key_Backtab.value: "tab",
        K.Key_Escape.value: "escape",
        K.Key_Backspace.value: "backspace",
        K.Key_Insert.value: "insert",
        K.Key_Delete.value: "delete",
        K.Key_Home.value: "home",
        K.Key_End.value: "end",
        K.Key_PageUp.value: "pageup",
        K.Key_PageDown.value: "pagedown",
        K.Key_Left.value: "left",
        K.Key_Right.value: "right",
        K.Key_Up.value: "up",
        K.Key_Down.value: "down",
        K.Key_Minus.value: "minus",
        K.Key_Equal.value: "equal",
        K.Key_BracketLeft.value: "bracketleft",
        K.Key_BracketRight.value: "bracketright",
        K.Key_Backslash.value: "backslash",
        K.Key_Semicolon.value: "semicolon",
        K.Key_Apostrophe.value: "quote",
        K.Key_QuoteLeft.value: "grave",
        K.Key_Comma.value: "comma",
        K.Key_Period.value: "period",
        K.Key_Slash.value: "slash",
    }
    for i in range(26):
        table[K.Key_A.value + i] = chr(ord("a") + i)
    for i in range(10):
        table[K.Key_0.value + i] = str(i)
    for i in range(24):
        table[K.Key_F1.value + i] = f"f{i + 1}"
    return table


_KEY_NAMES = _build_key_table()

# With Shift held Qt reports the shifted symbol ("!" for Shift+1). Global hotkeys are
# bound to physical keys, so map back to the unshifted key (US layout; on Windows
# letters and digits come from the layout-independent virtual key instead).
_SHIFTED_US: dict[int, str] = {
    Qt.Key.Key_Exclam.value: "1",
    Qt.Key.Key_At.value: "2",
    Qt.Key.Key_NumberSign.value: "3",
    Qt.Key.Key_Dollar.value: "4",
    Qt.Key.Key_Percent.value: "5",
    Qt.Key.Key_AsciiCircum.value: "6",
    Qt.Key.Key_Ampersand.value: "7",
    Qt.Key.Key_Asterisk.value: "8",
    Qt.Key.Key_ParenLeft.value: "9",
    Qt.Key.Key_ParenRight.value: "0",
    Qt.Key.Key_Underscore.value: "minus",
    Qt.Key.Key_Plus.value: "equal",
    Qt.Key.Key_BraceLeft.value: "bracketleft",
    Qt.Key.Key_BraceRight.value: "bracketright",
    Qt.Key.Key_Bar.value: "backslash",
    Qt.Key.Key_Colon.value: "semicolon",
    Qt.Key.Key_QuoteDbl.value: "quote",
    Qt.Key.Key_AsciiTilde.value: "grave",
    Qt.Key.Key_Less.value: "comma",
    Qt.Key.Key_Greater.value: "period",
    Qt.Key.Key_Question.value: "slash",
}


def _modifier_names(mods: Qt.KeyboardModifier, macos: bool) -> list[str]:
    """Canonical modifier names in ctrl, alt, shift, meta order.

    On macOS Qt reports the Command key as ``ControlModifier`` and the Control key
    as ``MetaModifier``; our names follow the physical keys (Command = meta).
    """
    KM = Qt.KeyboardModifier
    ctrl_flag, meta_flag = (
        (KM.MetaModifier, KM.ControlModifier)
        if macos
        else (
            KM.ControlModifier,
            KM.MetaModifier,
        )
    )
    names: list[str] = []
    if mods & ctrl_flag:
        names.append("ctrl")
    if mods & KM.AltModifier:
        names.append("alt")
    if mods & KM.ShiftModifier:
        names.append("shift")
    if mods & meta_flag:
        names.append("meta")
    return names


def modifier_preview(names: Sequence[str], macos: bool | None = None) -> str:
    """Held modifiers in the notation of :func:`format_hotkey`, e.g. ``"Ctrl+Win+…"``.

    Shown while a shortcut is being recorded. Built with ``format_hotkey`` so
    the preview and the finished shortcut name the keys alike (Win on Windows,
    Super on Linux, ⌃⌥⇧⌘ on macOS). Empty when nothing is held.
    """
    if not names:
        return ""
    if macos is None:
        macos = sys.platform == "darwin"
    # format_hotkey needs a key: F1 is valid with any modifier; drop its label.
    text = format_hotkey(Hotkey(frozenset(names), "f1"), macos=macos)
    prefix = text.removesuffix("F1")
    if prefix == text:  # unexpected label: fall back to the plain names
        prefix = "+".join(name.capitalize() for name in names) + "+"
    return prefix + "…"


def hotkey_text_from_event(event: QKeyEvent, macos: bool | None = None) -> str | None:
    """Settings-style hotkey text (``"ctrl+alt+p"``) for a key press.

    Returns ``None`` for a bare modifier press. The result is *not* validated: an
    unsupported key or a missing modifier yields text that :func:`parse_hotkey`
    rejects with a helpful message.
    """
    if macos is None:
        macos = sys.platform == "darwin"
    key = int(event.key())
    if key in _MODIFIER_KEYS or key == 0:
        return None
    mods = event.modifiers()
    names = _modifier_names(mods, macos)
    name: str | None = None
    if mods & Qt.KeyboardModifier.KeypadModifier and key not in (
        Qt.Key.Key_Return.value,
        Qt.Key.Key_Enter.value,
    ):
        # Keypad keys have their own virtual keys; "ctrl+alt+1" would never fire from
        # the keypad, so do not pretend it would.
        name = "keypad" + (QKeySequence(key).toString().lower() or str(key))
    elif sys.platform == "win32":
        vk = int(event.nativeVirtualKey())
        if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
            name = chr(vk).lower()
    if name is None:
        name = _KEY_NAMES.get(key)
    if name is None and "shift" in names:
        name = _SHIFTED_US.get(key)
    if name is None:
        name = QKeySequence(key).toString(QKeySequence.SequenceFormat.PortableText).lower()
        name = name or f"key{key}"
    return "+".join([*names, name])


class HotkeyEdit(QLineEdit):
    """Records a global shortcut: click it, then press the key combination.

    Esc (without modifiers) cancels recording and Backspace/Delete clears the
    shortcut. :meth:`hotkey` returns the canonical settings text (``"ctrl+alt+p"``)
    when the combination is valid, otherwise the raw attempt, so the owner can
    show why it was rejected.
    """

    hotkey_changed = Signal(str)
    #: True while the field waits for a key combination.
    recording_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setPlaceholderText("Not set")
        self.setToolTip("Click, then press the shortcut. Backspace clears it, Esc cancels.")
        self._value = ""
        self._recording = False
        self._invalid = False

    # --------------------------------------------------------------- value
    def hotkey(self) -> str:
        """Canonical hotkey text, ``""`` when unset, or the raw text if invalid."""
        return self._value

    def set_hotkey(self, text: str) -> None:
        """Set the shortcut (canonicalised when valid). Emits ``hotkey_changed``."""
        text = (text or "").strip()
        try:
            value = str(parse_hotkey(text)) if text else ""
        except ValueError:
            value = text
        changed = value != self._value
        self._value = value
        self._refresh()
        if changed:
            self.hotkey_changed.emit(value)

    def set_invalid(self, invalid: bool) -> None:
        """Mark the field as rejected (red frame)."""
        if invalid != self._invalid:
            self._invalid = invalid
            self.setStyleSheet(f"border: 1px solid {DANGER};" if invalid else "")

    @property
    def recording(self) -> bool:
        return self._recording

    def display_text(self) -> str:
        """What the field shows for the current value (platform notation)."""
        if not self._value:
            return ""
        try:
            return format_hotkey(parse_hotkey(self._value))
        except ValueError:
            return self._value

    # --------------------------------------------------------------- events
    def event(self, event: QEvent) -> bool:
        # While recording, claim every key combination before the dialog's own
        # shortcuts (button mnemonics, Esc/Enter defaults) can act on it.
        if event.type() == QEvent.Type.ShortcutOverride and self._recording:
            event.accept()
            return True
        return super().event(event)

    def focusInEvent(self, event: QFocusEvent) -> None:
        super().focusInEvent(event)
        self._set_recording(True)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self._set_recording(False)

    def hideEvent(self, event: QHideEvent) -> None:
        # A dialog closed while recording never delivers focusOut to the field.
        super().hideEvent(event)
        self._set_recording(False)

    def _set_recording(self, recording: bool) -> None:
        if recording == self._recording:
            return
        self._recording = recording
        self._refresh()
        self.recording_changed.emit(recording)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = int(event.key())
        bare = not _modifier_names(event.modifiers(), sys.platform == "darwin")
        if bare and key == Qt.Key.Key_Escape.value:
            self._refresh()
            self.clearFocus()
            return
        if bare and key in (Qt.Key.Key_Backspace.value, Qt.Key.Key_Delete.value):
            self.set_hotkey("")
            return
        text = hotkey_text_from_event(event)
        if text is None:
            # Only modifiers so far: show them as a live preview.
            held = _modifier_names(event.modifiers(), sys.platform == "darwin")
            self.setText(modifier_preview(held))
            return
        self.set_hotkey(text)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if int(event.key()) in _MODIFIER_KEYS:
            held = _modifier_names(event.modifiers(), sys.platform == "darwin")
            if not held:
                self._refresh()
            return
        super().keyReleaseEvent(event)

    def _refresh(self) -> None:
        self.setPlaceholderText("Press a shortcut…" if self._recording else "Not set")
        self.setText(self.display_text())


# ======================================================================= bindings
@dataclass
class _Binding:
    """Connects one settings key to a widget.

    ``snapshot`` captures the raw widget state; while it equals the state right
    after :meth:`load`, the originally loaded value is returned unchanged.
    """

    key: str
    widget: QWidget
    read: Callable[[], Any]
    write: Callable[[Any], None]
    snapshot: Callable[[], Any]
    loaded: Any = None
    loaded_snapshot: Any = None

    def load(self, value: Any) -> None:
        self.write(value)
        self.loaded = value
        self.loaded_snapshot = self.snapshot()

    @property
    def modified(self) -> bool:
        return bool(self.snapshot() != self.loaded_snapshot)

    def value(self) -> Any:
        return self.read() if self.modified else self.loaded


def _hotkey_writer(edit: HotkeyEdit) -> Callable[[Any], None]:
    def write(value: Any) -> None:
        edit.set_hotkey(str(value or ""))

    return write


def _get(settings: Settings, key: str) -> Any:
    section, name = key.split(".", 1)
    return getattr(getattr(settings, section), name)


def _set(settings: Settings, key: str, value: Any) -> None:
    section, name = key.split(".", 1)
    setattr(getattr(settings, section), name, value)


def _doc(key: str) -> str:
    return str(_META.get(key, {}).get("doc") or "")


# ======================================================================= the dialog
class SettingsDialog(QDialog):
    """All user settings, grouped into pages with a sidebar.

    Args:
        controller: The app's ``Controller`` (``settings``, ``apply_settings``,
            ``stats_changed``, ``settings_changed``, ``backend_info``, ``calibration``,
            ``platform``).
        parent: Optional parent widget.

    Signals:
        calibration_requested: The user pressed "Recalibrate…".
        setup_requested: The user pressed "Run setup assistant…" (the app opens
            the first-run assistant right away).
        applied: Settings were handed to the controller (argument: the new ``Settings``).
    """

    calibration_requested = Signal()
    setup_requested = Signal()
    applied = Signal(object)

    PAGES: tuple[tuple[str, str, str], ...] = (
        ("general", "General", "Startup, notifications and the vision backend."),
        ("switching", "Switching", "When and how the cursor follows your gaze."),
        (
            "presence",
            "Presence & privacy",
            "Walk-away lock, camera privacy and the shoulder guard.",
        ),
        ("camera", "Camera & performance", "Capture device, frame rate and CPU use."),
        ("hotkeys", "Hotkeys", "Global shortcuts that work in every application."),
        ("diagnostics", "Diagnostics", "System report for bug reports and troubleshooting."),
    )

    def __init__(self, controller: object, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._platform = platform_from_controller(controller)
        self._caps = _capabilities(self._platform)
        self._base = settings_from_controller(controller)
        self._bindings: dict[str, _Binding] = {}
        self._hotkey_edits: dict[str, HotkeyEdit] = {}
        self._hotkey_errors: dict[str, str] = {}
        #: layout_conflict() answers by canonical hotkey text.
        self._conflicts: dict[str, str | None] = {}
        #: The footer shows a hotkey error (cleared once the shortcuts are valid).
        self._status_shows_hotkey_error = False
        self._pages: dict[str, int] = {}
        self._diagnostics_loaded = False
        self._loading = False
        #: This dialog asked the controller to release the global hotkeys.
        self._hotkeys_suspended = False
        self._scale = ui_scale(self.screen() or QGuiApplication.primaryScreen())

        self.setWindowTitle(f"{APP_NAME} Settings")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(int(760 * self._scale), int(540 * self._scale))
        self.setSizeGripEnabled(True)

        self._camera_probe = CameraProbe(self)
        self._camera_probe.finished.connect(self._on_cameras_detected)
        #: Camera indices left out of the running probe (the one in use).
        self._probe_skip: set[int] = set()

        self._build()
        self._autostart_initial = self._read_autostart()
        self._autostart.setChecked(self._autostart_initial)
        self._refresh_autostart_note()
        self._load(self._base)

        self._connect_controller_signal("stats_changed", self._on_stats)
        self._connect_controller_signal("settings_changed", self._on_settings_changed)
        self._connect_controller_signal("state_changed", self._on_state_changed)
        self._refresh_calibration_status()
        self.show_page("general")

    # ============================================================== public API
    def settings(self) -> Settings:
        """The settings as currently shown (a new object; unexposed fields preserved)."""
        out = self._base.copy()
        for key, binding in self._bindings.items():
            _set(out, key, binding.value())
        return out

    def is_modified(self) -> bool:
        """Whether anything differs from the last loaded/applied state."""
        return self._autostart.isChecked() != self._autostart_initial or any(
            b.modified for b in self._bindings.values()
        )

    def is_valid(self) -> bool:
        return not self._hotkey_errors

    def bound_keys(self) -> list[str]:
        """Every ``section.field`` key that has a widget."""
        return list(self._bindings)

    def widget_for(self, key: str) -> QWidget:
        """The editor widget bound to ``key`` (for tests and focus handling)."""
        return self._bindings[key].widget

    def hotkey_edit(self, action: str) -> HotkeyEdit:
        return self._hotkey_edits[action]

    def hotkey_errors(self) -> dict[str, str]:
        return dict(self._hotkey_errors)

    def stats_text(self) -> str:
        return self._stats_label.text()

    def show_page(self, key: str) -> None:
        """Switch to a page by key (``"general"``, ``"hotkeys"``, …)."""
        row = self._pages.get(key)
        if row is not None:
            self._nav.setCurrentRow(row)

    def current_page(self) -> str:
        row = self._nav.currentRow()
        return next((k for k, r in self._pages.items() if r == row), "")

    def apply(self) -> bool:
        """Validate and hand the settings to the controller. Returns ``False`` if invalid."""
        if not self.is_valid():
            self._show_hotkey_error()
            return False
        new = self.settings()
        if new.to_dict() != self._base.to_dict():
            try:
                self._controller.apply_settings(new)  # type: ignore[attr-defined]
            except Exception as exc:
                log.exception("Applying settings failed")
                self._set_status(f"Could not apply the settings: {exc}", DANGER)
                return False
            self.applied.emit(new)
        # Reload from the controller so values it adjusted (clamped) are shown and the
        # new state becomes the baseline for "modified".
        current = new
        if hasattr(self._controller, "settings"):
            current = settings_from_controller(self._controller)
        autostart_ok = self._apply_autostart()
        self._load(current)
        if autostart_ok:
            self._set_status("", None)
        return autostart_ok

    def restore_defaults(self) -> None:
        """Show default values (still to be applied).

        Settings without a widget, such as the first-run flag, keep their value.
        """
        defaults = Settings()
        for key, binding in self._bindings.items():
            binding.write(_get(defaults, key))
        self._on_changed()

    # ============================================================== QDialog
    def accept(self) -> None:
        if self.apply():
            super().accept()

    def hideEvent(self, event: QHideEvent) -> None:
        # Closed or hidden while a shortcut field had focus: give the hotkeys
        # back now. The deferred resume of _on_hotkey_recording would never run
        # once the dialog is deleted.
        super().hideEvent(event)
        self._resume_hotkeys()

    # ============================================================== building
    def _build(self) -> None:
        s = self._scale
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        self._nav = QListWidget()
        self._nav.setObjectName("settingsNav")
        self._nav.setFrameShape(QFrame.Shape.NoFrame)
        self._nav.setFixedWidth(int(200 * s))
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._nav.setStyleSheet(
            f"""
            QListWidget#settingsNav {{
                padding: {int(10 * s)}px {int(8 * s)}px;
                background: palette(alternate-base);
                outline: 0;
            }}
            QListWidget#settingsNav::item {{
                padding: {int(8 * s)}px {int(10 * s)}px;
                margin: {int(1 * s)}px 0;
                border-radius: {int(6 * s)}px;
            }}
            QListWidget#settingsNav::item:hover:!selected {{
                background: rgba(127, 127, 127, 0.10);
            }}
            QListWidget#settingsNav::item:selected {{
                background: rgba(99, 102, 241, 0.18);
                color: palette(text);
                border-left: {max(2, int(3 * s))}px solid {ACCENT};
            }}
            """
        )
        body.addWidget(self._nav)

        self._stack = QStackedWidget()
        body.addWidget(self._stack, 1)

        builders: dict[str, Callable[[QVBoxLayout], None]] = {
            "general": self._build_general,
            "switching": self._build_switching,
            "presence": self._build_presence,
            "camera": self._build_camera,
            "hotkeys": self._build_hotkeys,
            "diagnostics": self._build_diagnostics,
        }
        for key, title, subtitle in self.PAGES:
            page, layout = self._page(title, subtitle)
            builders[key](layout)
            layout.addStretch(1)
            self._pages[key] = self._stack.addWidget(page)
            self._nav.addItem(QListWidgetItem(title))
        self._nav.currentRowChanged.connect(self._on_page_changed)

        # Footer: status line + buttons.
        footer = QFrame()
        footer.setFrameShape(QFrame.Shape.NoFrame)
        foot = QHBoxLayout(footer)
        foot.setContentsMargins(int(16 * s), int(10 * s), int(16 * s), int(14 * s))
        self._defaults_button = QPushButton("Restore defaults")
        self._defaults_button.setToolTip("Show the default value of every setting (Apply to keep).")
        self._defaults_button.clicked.connect(self.restore_defaults)
        foot.addWidget(self._defaults_button)
        self._status = QLabel()
        self._status.setWordWrap(True)
        foot.addWidget(self._status, 1)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._buttons.clicked.connect(self._on_button)
        foot.addWidget(self._buttons)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)
        root.addWidget(footer)

    def _page(self, title: str, subtitle: str) -> tuple[QScrollArea, QVBoxLayout]:
        s = self._scale
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(int(24 * s), int(20 * s), int(24 * s), int(16 * s))
        layout.setSpacing(int(14 * s))
        heading = QLabel(title)
        font = heading.font()
        font.setPointSizeF(font.pointSizeF() * 1.45)
        font.setWeight(font.Weight.DemiBold)
        heading.setFont(font)
        layout.addWidget(heading)
        sub = self._hint(subtitle)
        layout.addWidget(sub)
        scroll.setWidget(inner)
        return scroll, layout

    def _group(self, layout: QVBoxLayout, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(int(16 * self._scale))
        form.setVerticalSpacing(int(8 * self._scale))
        layout.addWidget(box)
        return form

    def _hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _banner(self, text: str) -> QFrame:
        s = self._scale
        frame = QFrame()
        frame.setObjectName("banner")
        frame.setStyleSheet(
            "QFrame#banner { background: rgba(99, 102, 241, 0.10);"
            " border: 1px solid rgba(99, 102, 241, 0.35);"
            f" border-radius: {int(8 * s)}px; }}"
        )
        row = QHBoxLayout(frame)
        row.setContentsMargins(int(14 * s), int(10 * s), int(14 * s), int(10 * s))
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        # Banners can name commands to paste into the desktop's shortcut settings.
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(label, 1)
        return frame

    @staticmethod
    def _row_label(text: str, tooltip: str) -> QLabel:
        label = QLabel(text)
        label.setToolTip(tooltip)
        return label

    # -------------------------------------------------------------- bind helpers
    def _register(self, binding: _Binding, signal: Any) -> _Binding:
        tooltip = _doc(binding.key)
        if tooltip and not binding.widget.toolTip():
            binding.widget.setToolTip(tooltip)
        self._bindings[binding.key] = binding
        signal.connect(self._on_changed)
        return binding

    def _check(self, form: QFormLayout, key: str, text: str) -> QCheckBox:
        box = QCheckBox(text)

        def read() -> bool:
            return box.isChecked()

        def write(value: Any) -> None:
            box.setChecked(bool(value))

        self._register(_Binding(key, box, read, write, box.isChecked), box.toggled)
        form.addRow(box)
        return box

    def _combo(
        self,
        form: QFormLayout | None,
        key: str,
        label: str,
        options: Sequence[tuple[str, str]],
    ) -> QComboBox:
        combo = QComboBox()
        for value, text in options:
            combo.addItem(text, value)
        # Settings may contain a value unknown to this UI version; keep it selectable.
        for value in _META.get(key, {}).get("choices") or ():
            if combo.findData(value) < 0:
                combo.addItem(value, value)

        def read() -> Any:
            return combo.currentData()

        def write(value: Any) -> None:
            index = combo.findData(value)
            if index < 0:
                combo.addItem(str(value), value)
                index = combo.count() - 1
            combo.setCurrentIndex(index)

        self._register(
            _Binding(key, combo, read, write, combo.currentIndex), combo.currentIndexChanged
        )
        if form is not None:
            form.addRow(self._row_label(label, _doc(key)), combo)
        return combo

    def _int(
        self,
        form: QFormLayout | None,
        key: str,
        label: str,
        *,
        suffix: str = "",
        step: int = 1,
    ) -> QSpinBox:
        spin = QSpinBox()
        meta = _META.get(key, {})
        lo, hi = meta.get("lo"), meta.get("hi")
        spin.setRange(int(lo) if lo is not None else 0, int(hi) if hi is not None else 1_000_000)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setAccelerated(True)

        def write(value: Any) -> None:
            spin.setValue(int(value))

        self._register(_Binding(key, spin, spin.value, write, spin.value), spin.valueChanged)
        if form is not None:
            form.addRow(self._row_label(label, _doc(key)), spin)
        return spin

    def _float(
        self,
        form: QFormLayout | None,
        key: str,
        label: str,
        *,
        factor: float = 1.0,
        decimals: int = 1,
        step: float = 0.1,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        meta = _META.get(key, {})
        lo, hi = meta.get("lo"), meta.get("hi")
        spin.setDecimals(decimals)
        spin.setRange(
            float(lo) * factor if lo is not None else 0.0,
            float(hi) * factor if hi is not None else 1e9,
        )
        spin.setSingleStep(step)
        spin.setSuffix(suffix)

        def read() -> float:
            return round(spin.value() / factor, decimals + 4)

        def write(value: Any) -> None:
            spin.setValue(float(value) * factor)

        self._register(_Binding(key, spin, read, write, spin.value), spin.valueChanged)
        if form is not None:
            form.addRow(self._row_label(label, _doc(key)), spin)
        return spin

    def _slider(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        *,
        left: str,
        right: str,
    ) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setPageStep(10)
        value_label = QLabel()
        value_label.setMinimumWidth(int(40 * self._scale))

        def show(v: int) -> None:
            value_label.setText(f"{v} %")

        slider.valueChanged.connect(show)

        def read() -> float:
            return slider.value() / 100.0

        def write(value: Any) -> None:
            slider.setValue(round(float(value) * 100))
            show(slider.value())

        self._register(_Binding(key, slider, read, write, slider.value), slider.valueChanged)
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        low = self._hint(left)
        low.setWordWrap(False)
        high = self._hint(right)
        high.setWordWrap(False)
        lay.addWidget(low)
        lay.addWidget(slider, 1)
        lay.addWidget(high)
        lay.addWidget(value_label)
        form.addRow(self._row_label(label, _doc(key)), row)
        return slider

    def _lines(self, form: QFormLayout, key: str, label: str, placeholder: str) -> QPlainTextEdit:
        edit = QPlainTextEdit()
        edit.setPlaceholderText(placeholder)
        edit.setTabChangesFocus(True)
        edit.setFixedHeight(int(84 * self._scale))

        def read() -> list[str]:
            return [line.strip() for line in edit.toPlainText().splitlines() if line.strip()]

        def write(value: Any) -> None:
            edit.setPlainText("\n".join(str(v) for v in (value or [])))

        self._register(_Binding(key, edit, read, write, edit.toPlainText), edit.textChanged)
        form.addRow(self._row_label(label, _doc(key)), edit)
        return edit

    @staticmethod
    def _depends(master: QCheckBox, widgets: Sequence[QWidget]) -> None:
        def update(checked: bool) -> None:
            for widget in widgets:
                widget.setEnabled(checked)

        master.toggled.connect(update)
        update(master.isChecked())

    # -------------------------------------------------------------- pages
    def _build_general(self, layout: QVBoxLayout) -> None:
        form = self._group(layout, "Startup")
        self._autostart = QCheckBox(f"Start {APP_NAME} when I log in")
        supported = self._autostart_supported()
        self._autostart.setEnabled(supported)
        self._autostart.setToolTip(
            "Runs quietly in the tray after you log in."
            if supported
            else "Start at login is not supported on this system."
        )
        self._autostart.toggled.connect(self._on_changed)
        form.addRow(self._autostart)
        self._autostart_note = self._hint("")
        self._autostart_note.setVisible(False)
        form.addRow(self._autostart_note)
        self._check(form, "general.start_paused", "Start with tracking paused")
        # Opens the assistant now. A "show it at next start" switch would have
        # to clear general.first_run_done, and until the assistant is finished
        # that flag also turns the walk-away lock into a mere notification.
        setup = QPushButton("Run setup assistant…")
        setup.setToolTip(
            "Check the camera and choose what happens when you walk away, step by step. "
            "Cancelling it changes nothing."
        )
        setup.clicked.connect(self.setup_requested)
        form.addRow(setup)

        form = self._group(layout, "Interface")
        self._check(form, "general.notifications", "Show notifications")
        self._check(form, "ui.show_gaze_overlay", "Show a dot where I am looking (testing aid)")

        form = self._group(layout, "Calibration")
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        self._calibration_status = QLabel()
        self._calibration_status.setWordWrap(True)
        lay.addWidget(self._calibration_status, 1)
        recalibrate = QPushButton("Recalibrate…")
        recalibrate.setToolTip("Run the calibration again (about 30 seconds).")
        recalibrate.clicked.connect(self.calibration_requested)
        lay.addWidget(recalibrate)
        form.addRow(row)

        form = self._group(layout, "Advanced")
        available = self._available_backends()
        options = []
        for value, label in BACKEND_LABELS.items():
            missing = available is not None and value != "auto" and value not in available
            options.append((value, f"{label} — not available" if missing else label))
        self._combo(form, "general.backend", "Vision backend", options)
        form.addRow(
            self._hint(
                "Face mesh follows your head and your eyes; Lite follows only the head, "
                "for slow computers. Changing the backend requires a new calibration."
            )
        )
        self._combo(
            form,
            "general.log_level",
            "Log level",
            [("DEBUG", "Debug"), ("INFO", "Info"), ("WARNING", "Warning"), ("ERROR", "Error")],
        )

    def _build_switching(self, layout: QVBoxLayout) -> None:
        form = self._group(layout, "Switching")
        enabled = self._check(form, "switching.enabled", "Move the cursor to the monitor I look at")
        target = self._combo(
            form,
            "switching.cursor_target",
            "Cursor lands",
            [
                ("last", "Where it was last on that monitor"),
                ("center", "In the centre of the monitor"),
                ("gaze", "Where I am looking (estimated)"),
            ],
        )
        focus = self._check(
            form, "switching.focus_window", "Also give keyboard focus to that monitor"
        )
        if not self._caps.get("focus", True):
            focus.setText(focus.text() + " (not supported on this system)")

        timing = self._group(layout, "Timing")
        dwell = self._int(timing, "switching.dwell_ms", "Look for at least", suffix=" ms", step=50)
        cooldown = self._int(
            timing, "switching.cooldown_ms", "Pause between switches", suffix=" ms", step=100
        )
        mouse = self._int(
            timing, "switching.mouse_grace_ms", "Wait after mouse use", suffix=" ms", step=250
        )
        typing = self._int(
            timing, "switching.typing_grace_ms", "Wait after typing", suffix=" ms", step=250
        )
        reading = self._int(
            timing,
            "switching.reading_grace_ms",
            "… while reading another monitor",
            suffix=" ms",
            step=500,
        )

        tuning = self._group(layout, "Sensitivity")
        smoothing = self._slider(
            tuning, "switching.smoothing", "Smoothing", left="Responsive", right="Steady"
        )
        hysteresis = self._float(
            tuning,
            "switching.hysteresis",
            "Cross into a monitor by",
            factor=100.0,
            decimals=0,
            step=1.0,
            suffix=" % of its size",
        )
        margin = self._float(
            tuning,
            "switching.off_screen_margin",
            "Ignore gaze beyond",
            factor=100.0,
            decimals=0,
            step=5.0,
            suffix=" % of the diagonal",
        )
        self._depends(
            enabled,
            [target, focus, dwell, cooldown, mouse, typing, reading, smoothing, hysteresis, margin],
        )

        panes = self._group(layout, "Split panes (experimental)")
        follow = self._check(
            panes, "panes.enabled", "Also move keyboard focus to the split pane I look at"
        )
        if not self._caps.get("panes", True):
            follow.setText(follow.text() + " (not supported on this system)")
        move = self._check(
            panes, "panes.move_cursor", "Also move the cursor into that pane, where I left it"
        )
        precision = self._float(
            panes,
            "panes.precision",
            "Only panes at least",
            decimals=1,
            step=0.5,
            suffix=" × my gaze error",
        )
        panes.addRow(
            self._hint(
                "Works with tmux, WezTerm and Windows Terminal panes on the monitor you are on. "
                "Small panes are left alone: how small depends on how accurate your calibration "
                "is."
            )
        )
        desktop = self._check(
            panes,
            "panes.desktop_apps",
            "Also switch between side-by-side sessions in the Claude and ChatGPT apps",
        )
        desktop_hint = self._hint(
            "Two Claude chats, or a ChatGPT or Codex conversation and its side chat: the one you "
            "look at gets the focus in its message box. Windows only. Asks the app for its "
            "accessibility tree, which can cost the app some CPU and memory."
        )
        panes.addRow(desktop_hint)
        self._depends(follow, [move, precision, desktop, desktop_hint])

        learn = self._group(layout, "Adaptive accuracy")
        adaptive = self._check(learn, "learning.adaptive", "Learn from how I use the mouse")
        max_samples = self._int(
            learn, "learning.max_samples", "Keep at most", suffix=" samples", step=50
        )
        self._depends(adaptive, [max_samples])
        self._check(learn, "learning.drift_alerts", "Suggest recalibrating when accuracy drops")

    def _build_presence(self, layout: QVBoxLayout) -> None:
        layout.addWidget(
            self._banner(
                "<b>Everything stays on this computer.</b> Camera frames are analysed in "
                "memory and never saved or sent anywhere. Privacy mode turns the camera off "
                "completely."
            )
        )
        form = self._group(layout, "When I walk away")
        enabled = self._check(form, "presence.enabled", "React when I leave the computer")
        action = self._combo(
            form,
            "presence.action",
            "Then",
            [
                ("lock", "Lock the computer"),
                ("lock_and_display_off", "Lock and turn the displays off"),
                ("display_off", "Turn the displays off"),
                ("notify", "Only show a notification"),
                ("none", "Do nothing"),
            ],
        )
        self._action_note = self._hint("")
        self._action_note.setVisible(False)
        form.addRow(self._action_note)
        action.currentIndexChanged.connect(self._update_action_note)
        timeout = self._int(form, "presence.away_timeout_s", "After", suffix=" s away", step=5)
        warning = self._int(form, "presence.warning_s", "Countdown", suffix=" s", step=1)
        idle = self._check(
            form, "presence.require_input_idle", "Keyboard or mouse use counts as being here"
        )
        if not self._caps.get("input_idle", True):
            # Wayland outside GNOME: only the camera can cancel the countdown.
            idle.setText(idle.text() + " (not supported on this system)")
        wake = self._check(
            form, "presence.wake_on_return", "Turn the displays back on when I return"
        )
        self._depends(enabled, [action, timeout, warning, idle, wake])

        form = self._group(layout, "Camera privacy")
        self._check(
            form,
            "privacy.remember_privacy_mode",
            "Keep privacy mode on after a restart (the camera stays off)",
        )
        self._check(
            form, "privacy.pause_when_locked", "Release the camera while the screen is locked"
        )
        yield_box = self._check(
            form, "privacy.yield_camera", "Release the camera while another app uses it"
        )
        if not self._caps.get("camera_in_use", True):
            yield_box.setText(yield_box.text() + " (not supported on this system)")
        self._lines(
            form,
            "privacy.pause_for_apps",
            "Pause while these apps run",
            "One program per line, e.g.\nzoom.exe\nobs64.exe",
        )

        form = self._group(layout, "Shoulder guard")
        guard = self._check(
            form, "privacy.shoulder_guard", "React when someone looks over my shoulder"
        )
        guard_action = self._combo(
            form,
            "privacy.guard_action",
            "Then",
            [
                ("curtain", "Cover the screens"),
                ("notify", "Only show a notification"),
                ("lock", "Lock the computer"),
            ],
        )
        delay = self._float(
            form, "privacy.guard_delay_s", "After", decimals=1, step=0.5, suffix=" s"
        )
        self._depends(guard, [guard_action, delay])

    def _build_camera(self, layout: QVBoxLayout) -> None:
        form = self._group(layout, "Camera")
        self._device = QComboBox()
        self._device.setEditable(True)
        self._device.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._device.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        device_line = self._device.lineEdit()
        if device_line is not None:
            device_line.setPlaceholderText("Camera number or a video file")

        def read_device() -> str:
            return self._device_value()

        def write_device(value: Any) -> None:
            self._select_device(str(value))

        self._register(
            _Binding("camera.device", self._device, read_device, write_device, read_device),
            self._device.currentTextChanged,
        )
        device_row = QWidget()
        lay = QHBoxLayout(device_row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._device, 1)
        self._detect_button = QPushButton("Detect cameras")
        self._detect_button.setToolTip("Look for connected cameras (each may briefly turn on).")
        self._detect_button.clicked.connect(self.detect_cameras)
        lay.addWidget(self._detect_button)
        browse = QToolButton()
        browse.setText("…")
        browse.setToolTip("Use a video or image file instead of a camera (for testing).")
        browse.clicked.connect(self._browse_video)
        lay.addWidget(browse)
        form.addRow(self._row_label("Device", _doc("camera.device")), device_row)

        size_row = QWidget()
        lay = QHBoxLayout(size_row)
        lay.setContentsMargins(0, 0, 0, 0)
        width = self._int(None, "camera.width", "", suffix=" px", step=160)
        height = self._int(None, "camera.height", "", suffix=" px", step=120)
        lay.addWidget(width, 1)
        lay.addWidget(QLabel("×"))
        lay.addWidget(height, 1)
        form.addRow(self._row_label("Resolution", "Requested capture size."), size_row)
        form.addRow(self._hint("640 × 480 is plenty: frames are scaled down before analysis."))
        self._combo(
            form,
            "camera.api",
            "Capture API",
            [
                ("auto", "Automatic"),
                ("dshow", "DirectShow (Windows)"),
                ("msmf", "Media Foundation (Windows)"),
                ("avfoundation", "AVFoundation (macOS)"),
                ("v4l2", "Video4Linux2 (Linux)"),
                ("any", "Any available"),
            ],
        )
        self._camera_note = self._hint("")
        self._camera_note.setVisible(False)
        form.addRow(self._camera_note)

        form = self._group(layout, "Performance")
        options = []
        profile_text = {
            "eco": "Eco — lowest CPU",
            "balanced": "Balanced (recommended)",
            "responsive": "Responsive — fastest switching",
        }
        try:
            from ..engine.scheduler import PROFILES
        except Exception:  # pragma: no cover - engine is part of the package
            PROFILES = {}
        for value in ("eco", "balanced", "responsive"):
            text = profile_text[value]
            fps = PROFILES.get(value, {}).get("active")
            if fps:
                text += f", up to {fps:g} fps"
            options.append((value, text))
        self._combo(form, "performance.profile", "Profile", options)
        gate = self._check(
            form, "performance.motion_gate", "Skip analysis while the picture is still"
        )
        threshold = self._float(
            form, "performance.motion_threshold", "Motion threshold", decimals=1, step=0.5
        )
        self._depends(gate, [threshold])

        form = self._group(layout, "Live")
        self._stats_label = QLabel("Waiting for statistics…")
        self._stats_label.setWordWrap(True)
        self._stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._stats_label)
        self._backend_label = self._hint("")
        form.addRow(self._backend_label)
        self._refresh_backend_label()

    def _on_hotkey_recording(self, recording: bool) -> None:
        # Registered global hotkeys are swallowed by the OS before they reach this
        # dialog, so they are released while a shortcut is being recorded.
        if recording:
            self._suspend_hotkeys(True)
            return
        # Moving from one shortcut field to the next leaves the first (False)
        # just before entering the second (True). Resuming at once would
        # re-register every hotkey (repeating "Hotkey unavailable" for one that
        # fails) only to release them again, so it waits until focus settled.
        QTimer.singleShot(0, self, self._resume_hotkeys_if_idle)

    def _resume_hotkeys_if_idle(self) -> None:
        """Give the hotkeys back unless a shortcut field is recording again."""
        if not any(edit.recording for edit in self._hotkey_edits.values()):
            self._resume_hotkeys()

    def _resume_hotkeys(self) -> None:
        """Give the hotkeys back now if this dialog released them."""
        self._suspend_hotkeys(False)

    def _suspend_hotkeys(self, suspended: bool) -> None:
        if suspended == self._hotkeys_suspended:
            return
        self._hotkeys_suspended = suspended
        suspend = getattr(self._controller, "suspend_hotkeys", None)
        if callable(suspend):
            try:
                suspend(suspended)
            except Exception:
                log.debug("Suspending hotkeys failed", exc_info=True)

    def _build_hotkeys(self, layout: QVBoxLayout) -> None:
        form = self._group(layout, "Global shortcuts")
        enabled = self._check(form, "hotkeys.enabled", "Enable global shortcuts")
        edits: list[QWidget] = []
        for action, label in HOTKEY_ACTIONS:
            key = f"hotkeys.{action}"
            edit = HotkeyEdit()
            edit.setToolTip(f"{_doc(key)}\nClick, then press the shortcut. Backspace clears it.")
            edit.recording_changed.connect(self._on_hotkey_recording)
            self._hotkey_edits[action] = edit
            self._register(
                _Binding(key, edit, edit.hotkey, _hotkey_writer(edit), edit.hotkey),
                edit.hotkey_changed,
            )
            clear = QToolButton()
            clear.setText("Clear")
            clear.setToolTip("Remove this shortcut.")
            clear.clicked.connect(lambda _=False, e=edit: e.set_hotkey(""))
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(edit, 1)
            lay.addWidget(clear)
            form.addRow(self._row_label(label, _doc(key)), row)
            edits.append(row)
        self._hotkey_error = QLabel()
        self._hotkey_error.setWordWrap(True)
        self._hotkey_error.setStyleSheet(f"color: {DANGER};")
        self._hotkey_error.setVisible(False)
        form.addRow(self._hotkey_error)
        # Valid shortcuts the OS will refuse (they type a character, e.g. AltGr).
        self._hotkey_warning = QLabel()
        self._hotkey_warning.setWordWrap(True)
        self._hotkey_warning.setStyleSheet(f"color: {WARNING};")
        self._hotkey_warning.setVisible(False)
        form.addRow(self._hotkey_warning)
        self._depends(enabled, edits)

        tip = (
            "Shortcuts need at least one of Ctrl, Alt or "
            + ("Cmd" if sys.platform == "darwin" else "Win" if sys.platform == "win32" else "Super")
            + ". Shift alone only works with function keys."
        )
        form.addRow(self._hint(tip))
        # The controller's manager knows best (e.g. XWayland limits); before it
        # exists fall back to the platform's capability matrix.
        manager = getattr(self._controller, "hotkey_manager", None)
        supported = bool(getattr(manager, "supported", self._caps.get("hotkeys", True)))
        note = getattr(manager, "note", None)
        note = note if isinstance(note, str) and note.strip() else None
        if not supported:
            note = note or "Global shortcuts are not available in this desktop session."
            text = html.escape(note)
            # Named as this copy is run (the .AppImage file, eye-tracker-cli, …):
            # the packages put no "eye-tracker" command on the PATH, and a
            # desktop shortcut bound to a missing command fails silently. Added
            # unless the note already names the commands that way.
            if cli_command_text("ctl") not in note:
                commands = " · ".join(
                    f"<code>{html.escape(cli_command_text('ctl', action))}</code>"
                    for action in ("toggle", "privacy-toggle", "calibrate")
                )
                text += (
                    "<br>Bind these commands in your desktop's keyboard settings instead: "
                    + commands
                )
            layout.addWidget(self._banner(text))
        elif note:
            layout.addWidget(self._banner(html.escape(note)))

    def _build_diagnostics(self, layout: QVBoxLayout) -> None:
        self._diagnostics = QPlainTextEdit()
        self._diagnostics.setReadOnly(True)
        self._diagnostics.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._diagnostics.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._diagnostics.setMinimumHeight(int(300 * self._scale))
        self._diagnostics.setPlaceholderText("Collecting…")
        layout.addWidget(self._diagnostics, 1)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_diagnostics)
        copy = QPushButton("Copy report")
        copy.setToolTip("Copy the report to paste into a bug report.")
        copy.clicked.connect(self._copy_diagnostics)
        logs = QPushButton("Open log folder")
        logs.clicked.connect(self._open_log_folder)
        row.addWidget(refresh)
        row.addWidget(copy)
        row.addStretch(1)
        row.addWidget(logs)
        layout.addLayout(row)
        layout.addWidget(self._hint("The report contains no images and nothing is sent anywhere."))

    # ============================================================== loading
    def _load(self, settings: Settings) -> None:
        self._loading = True
        try:
            self._base = settings.copy()
            for key, binding in self._bindings.items():
                binding.load(_get(settings, key))
            # The start-at-login baseline is not reset here: it is what the OS
            # has (read at start, updated only by a successful change), so a
            # change that failed stays pending and OK/Apply retry it.
        finally:
            self._loading = False
        self._on_changed()

    # ============================================================== slots
    def _on_changed(self, *_args: object) -> None:
        if self._loading:
            return
        self._validate_hotkeys()
        self._update_action_note()
        valid = self.is_valid()
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        apply_button = self._buttons.button(QDialogButtonBox.StandardButton.Apply)
        if ok is not None:
            ok.setEnabled(valid)
        if apply_button is not None:
            apply_button.setEnabled(valid and self.is_modified())
        # The error label is on the Hotkeys page; the footer says why OK is off
        # whichever page is shown.
        if not valid:
            self._show_hotkey_error()
        elif self._status_shows_hotkey_error:
            self._set_status("", None)

    def _show_hotkey_error(self) -> None:
        """Put the first shortcut error in the footer (cleared once all are valid)."""
        self._set_status(
            next(iter(self._hotkey_errors.values()), "Fix the highlighted shortcut first."),
            DANGER,
        )
        self._status_shows_hotkey_error = True

    def _on_button(self, button: QAbstractButton) -> None:
        if self._buttons.buttonRole(button) == QDialogButtonBox.ButtonRole.ApplyRole:
            self.apply()

    def _on_page_changed(self, row: int) -> None:
        self._stack.setCurrentIndex(row)
        if row == self._pages.get("diagnostics") and not self._diagnostics_loaded:
            self.refresh_diagnostics()

    def _on_stats(self, stats: object) -> None:
        text = format_stats(stats)
        if text:
            self._stats_label.setText(text)

    def _on_state_changed(self, _state: object) -> None:
        # Calibration saved or invalidated (e.g. new monitor layout).
        self._refresh_calibration_status()

    def _on_settings_changed(self, settings: object) -> None:
        """Settings changed elsewhere: refresh the widgets the user has not touched."""
        if not isinstance(settings, Settings):
            return
        self._loading = True
        try:
            self._base = settings.copy()
            for key, binding in self._bindings.items():
                if not binding.modified:
                    binding.load(_get(settings, key))
        finally:
            self._loading = False
        self._on_changed()
        self._refresh_calibration_status()
        self._refresh_backend_label()

    # ============================================================== hotkeys
    def _validate_hotkeys(self) -> None:
        errors: dict[str, str] = {}
        warnings: dict[str, str] = {}
        seen: dict[str, str] = {}
        labels = dict(HOTKEY_ACTIONS)
        enabled = self._bindings.get("hotkeys.enabled")
        # With global shortcuts off nothing is registered, and the fields cannot
        # be edited: a bad stored value must not block every other change.
        active = enabled is None or bool(enabled.read())
        for action, edit in self._hotkey_edits.items():
            text = edit.hotkey()
            if not text or not active:
                continue
            try:
                canonical = str(parse_hotkey(text))
            except ValueError as exc:
                errors[action] = f"{labels[action]}: {exc}"
                continue
            if canonical in seen:
                errors[action] = f"{labels[action]}: same shortcut as {labels[seen[canonical]]}"
                continue
            seen[canonical] = action
            conflict = self._layout_conflict(canonical)
            if conflict:
                warnings[action] = f"{labels[action]}: {conflict}. Choose another shortcut."
        self._hotkey_errors = errors
        for action, edit in self._hotkey_edits.items():
            edit.set_invalid(action in errors or action in warnings)
        self._hotkey_error.setText("\n".join(errors.values()))
        self._hotkey_error.setVisible(bool(errors))
        self._hotkey_warning.setText("\n".join(warnings.values()))
        self._hotkey_warning.setVisible(bool(warnings))

    def _layout_conflict(self, hotkey: str) -> str | None:
        """Why ``hotkey`` would swallow a character on an installed keyboard layout.

        Asks the controller's hotkey manager (``layout_conflict``, read-only);
        answers are cached for the life of the dialog, since the check reads
        every installed layout.
        """
        if hotkey not in self._conflicts:
            manager = getattr(self._controller, "hotkey_manager", None)
            check = getattr(manager, "layout_conflict", None)
            result: str | None = None
            if callable(check):
                try:
                    value = check(hotkey)
                    result = str(value) if value else None
                except Exception:
                    log.debug("layout_conflict(%s) failed", hotkey, exc_info=True)
            self._conflicts[hotkey] = result
        return self._conflicts[hotkey]

    # ============================================================== camera
    def detect_cameras(self) -> None:
        """Probe connected cameras in the background and fill the device list."""
        api = self._bindings["camera.api"].value() or "auto"
        self._probe_skip = camera_in_use(self._controller)
        if self._camera_probe.start(str(api), skip=self._probe_skip):
            self._detect_button.setEnabled(False)
            self._detect_button.setText("Detecting…")

    def _on_cameras_detected(self, cameras: object) -> None:
        self._detect_button.setEnabled(True)
        self._detect_button.setText("Detect cameras")
        current = self._device_value()
        found = list(cameras) if isinstance(cameras, list) else []
        with _blocked(self._device):
            self._device.clear()
            for info in found:
                index = getattr(info, "index", None)
                if index is None:
                    continue
                w, h = getattr(info, "width", 0), getattr(info, "height", 0)
                size = f" — {w}×{h}" if w and h else ""
                self._device.addItem(
                    f"{getattr(info, 'name', f'Camera {index}')}{size}", str(index)
                )
            for index in sorted(self._probe_skip):
                # Not probed, but it exists: keep it selectable.
                if self._device.findData(str(index)) < 0:
                    self._device.addItem(f"Camera {index} (in use)", str(index))
        self._select_device(current)
        other = "other " if self._probe_skip else ""
        parts = [f"Found {len(found)} {other}camera(s)." if found else f"No {other}camera found."]
        if self._probe_skip:
            parts.append("The camera in use is not probed.")
        self._camera_note.setText(" ".join(parts))
        self._camera_note.setVisible(True)
        self._on_changed()

    def _device_value(self) -> str:
        text = self._device.currentText().strip()
        index = self._device.currentIndex()
        if index >= 0 and self._device.itemText(index) == self._device.currentText():
            data = self._device.itemData(index)
            if isinstance(data, str) and data:
                return data
        return text

    def _select_device(self, value: str) -> None:
        with _blocked(self._device):
            index = self._device.findData(value)
            if index < 0:
                label = f"Camera {value}" if value.isdigit() else value
                self._device.addItem(label, value)
                index = self._device.count() - 1
            self._device.setCurrentIndex(index)
        # currentTextChanged was blocked; recompute the dirty state explicitly.
        self._on_changed()

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a video or image",
            "",
            "Video or image (*.mp4 *.avi *.mkv *.mov *.webm *.png *.jpg *.jpeg *.bmp)"
            ";;All files (*)",
        )
        if path:
            self._select_device(path)

    # ============================================================== autostart
    @staticmethod
    def _autostart_supported() -> bool:
        try:
            return bool(autostart.is_supported())
        except Exception:
            return False

    @staticmethod
    def _read_autostart() -> bool:
        try:
            return bool(autostart.is_enabled())
        except Exception:
            log.debug("autostart.is_enabled failed", exc_info=True)
            return False

    def _apply_autostart(self) -> bool:
        wanted = self._autostart.isChecked()
        if wanted == self._autostart_initial:
            return True
        try:
            if wanted:
                autostart.enable()
            else:
                autostart.disable()
        except autostart.AutostartError as exc:
            # _autostart_initial keeps the OS state, so the change stays pending
            # (Apply enabled) and the next OK/Apply tries again.
            log.warning("Changing start at login failed: %s", exc)
            self._set_status(str(exc), DANGER)
            return False
        self._autostart_initial = wanted
        self._refresh_autostart_note()
        return True

    def _refresh_autostart_note(self) -> None:
        """Explain an unchecked box whose login item exists but does not start this copy."""
        status = util.autostart_status(autostart) if self._autostart_supported() else ""
        text = _AUTOSTART_NOTES.get(status, "")
        self._autostart_note.setText(text)
        self._autostart_note.setVisible(bool(text))

    # ============================================================== misc
    def _connect_controller_signal(self, name: str, slot: Callable[[object], None]) -> None:
        signal = getattr(self._controller, name, None)
        if signal is None:
            return
        try:
            signal.connect(slot)
        except (AttributeError, TypeError, RuntimeError):
            log.debug("Controller signal %s not connectable", name, exc_info=True)

    def _set_status(self, text: str, color: str | None) -> None:
        self._status_shows_hotkey_error = False
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color};" if color else "")

    def _available_backends(self) -> list[str] | None:
        try:
            from ..vision.backends import available_backends

            return list(available_backends())
        except Exception:
            log.debug("available_backends failed", exc_info=True)
            return None

    def _backend_info(self) -> tuple[str, str] | None:
        getter = getattr(self._controller, "backend_info", None)
        if getter is None:
            return None
        try:
            name, version = getter()
            return str(name), str(version)
        except Exception:
            return None

    def _refresh_backend_label(self) -> None:
        info = self._backend_info()
        self._backend_label.setText(f"Backend in use: {info[0]} ({info[1]})" if info else "")
        self._backend_label.setVisible(bool(info))

    def _refresh_calibration_status(self) -> None:
        getter = getattr(self._controller, "calibration", None)
        data = None
        if callable(getter):
            try:
                data = getter()
            except Exception:
                log.debug("controller.calibration() failed", exc_info=True)
        if data is None:
            self._calibration_status.setText("Not calibrated yet.")
            return
        grade = getattr(data, "grade", None)
        created = str(getattr(data, "created_at", "") or "")[:10]
        parts = ["Calibrated"]
        if created:
            parts.append(f"on {created}")
        text = " ".join(parts)
        if grade:
            text += f" · quality: {grade}"
        text += "."
        # A stored calibration may no longer fit (new monitor layout, other backend).
        if getattr(self._controller, "is_calibrated", True) is False:
            reason = getattr(self._controller, "calibration_reason", "")
            if isinstance(reason, str) and reason:
                text += f" Needs recalibration: {reason}."
            else:
                text += " Needs recalibration."
        self._calibration_status.setText(text)

    def _update_action_note(self, *_args: object) -> None:
        binding = self._bindings.get("presence.action")
        if binding is None:
            return
        action = binding.read()
        needs = {
            "lock": ["lock"],
            "display_off": ["display_off"],
            "lock_and_display_off": ["lock", "display_off"],
        }.get(str(action), [])
        missing = [n for n in needs if not self._caps.get(n, True)]
        if missing:
            what = " and ".join(
                "locking" if n == "lock" else "turning displays off" for n in missing
            )
            self._action_note.setText(
                f"This system does not support {what}; a notification is shown instead."
            )
        self._action_note.setVisible(bool(missing))

    def refresh_diagnostics(self) -> None:
        """(Re)build the diagnostics report."""
        self._diagnostics_loaded = True
        # The report says which hotkeys are registered. Clicking from a shortcut
        # field to this page has only scheduled giving them back (see
        # _on_hotkey_recording); do it first, or every hotkey would look refused.
        self._resume_hotkeys_if_idle()
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            text = load_diagnostics_text(self._controller)
        except Exception as exc:
            log.warning("Diagnostics report failed", exc_info=True)
            text = f"Could not collect the diagnostics report: {exc}"
        finally:
            QGuiApplication.restoreOverrideCursor()
        self._diagnostics.setPlainText(text)

    def diagnostics_text(self) -> str:
        return self._diagnostics.toPlainText()

    def _copy_diagnostics(self) -> None:
        if not self._diagnostics_loaded:
            self.refresh_diagnostics()
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._diagnostics.toPlainText())
            self._set_status("Report copied to the clipboard.", None)

    def _open_log_folder(self) -> None:
        try:
            folder = paths.log_dir()
        except OSError as exc:
            self._set_status(f"Log folder unavailable: {exc}", DANGER)
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._set_status(f"Logs are in {folder}", None)


@contextlib.contextmanager
def _blocked(widget: QObject) -> Iterator[None]:
    """Block a widget's signals for the duration of the ``with`` block."""
    previous = widget.blockSignals(True)
    try:
        yield
    finally:
        widget.blockSignals(previous)
