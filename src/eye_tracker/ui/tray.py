"""System-tray icon, menu, tooltip and notifications.

The tray is the app's main surface. It mirrors the controller's state (icon,
tooltip, menu texts and check marks) and turns menu clicks into controller calls
or into signals the application wires to windows (settings, calibration,
preview, about, quit). It holds no logic of its own.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from .. import APP_NAME
from ..config import Settings
from ..platform.hotkeys import Hotkey, format_hotkey
from ..types import TrackingState
from . import icons, util

log = logging.getLogger(__name__)

__all__ = ["AutostartBackend", "TrayIcon"]

#: Windows truncates tray tooltips at 127 characters.
_TOOLTIP_LIMIT = 127
#: Longest detail (camera error, yield reason) appended to the menu's status line.
_STATUS_DETAIL_LIMIT = 60

_MUTED_GREY = "#9CA3AF"
_STATUS_COLORS: dict[TrackingState, str] = {
    TrackingState.STARTING: util.ACCENT,
    TrackingState.NEEDS_CALIBRATION: util.WARNING,
    TrackingState.CALIBRATING: util.ACCENT,
    TrackingState.TRACKING: util.SUCCESS,
    TrackingState.PAUSED: _MUTED_GREY,
    TrackingState.PRIVACY: util.ACCENT_LIGHT,
    TrackingState.AWAY: _MUTED_GREY,
    TrackingState.LOCKED: _MUTED_GREY,
    TrackingState.YIELDED: _MUTED_GREY,
    TrackingState.CAMERA_ERROR: util.DANGER,
}

#: Controller hotkey actions (``settings.hotkeys`` field names) per menu item.
_HOTKEY_ACTIONS = ("toggle_tracking", "toggle_privacy", "recalibrate")

_PAUSE_TIP = "Stop moving the cursor; the camera is released"
_RESUME_TIP = "Start following your gaze again"
_PRIVACY_TIP = "Release the camera completely (its light turns off)"
_CALIBRATE_TIP = "Look at a few dots so the tracker learns your monitors"


class AutostartBackend(Protocol):
    """What the tray needs from :mod:`eye_tracker.platform.autostart` (tests fake it)."""

    def is_supported(self) -> bool: ...
    def is_enabled(self) -> bool: ...
    def enable(self, background: bool = True) -> None: ...
    def disable(self) -> None: ...


def _default_autostart() -> AutostartBackend:
    from ..platform import autostart

    return autostart


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class TrayIcon(QObject):
    """Tray icon with the app menu.

    Menu: status line · Pause/Resume tracking · Privacy mode · Calibrate… ·
    Show gaze dot · Camera preview… · Settings… · Start at login · About · Quit.
    Global hotkeys are shown next to the actions they trigger. Double-clicking
    the icon opens the settings; a middle click pauses or resumes tracking.

    Args:
        controller: The :class:`~eye_tracker.engine.controller.Controller` (or a
            fake with the same signals, properties and methods).
        parent: Qt parent.
        autostart: Start-at-login backend; defaults to
            :mod:`eye_tracker.platform.autostart`.
        clock: Monotonic time source; only used by :meth:`mute_notifications`.
    """

    open_settings = Signal()
    open_calibration = Signal()
    open_preview = Signal()
    open_about = Signal()
    quit_requested = Signal()

    NOTIFICATION_MS = 6000

    def __init__(
        self,
        controller: Any,
        parent: QObject | None = None,
        *,
        autostart: AutostartBackend | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._autostart = autostart if autostart is not None else _default_autostart()
        self._clock = clock
        self._settings: Settings = util.controller_settings(controller)
        self._state: TrackingState = util.controller_state(controller)
        self._stats: dict[str, Any] = {}
        self._muted_until = -math.inf
        self._icon_key: tuple[object, ...] | None = None
        self._hotkey_labels: dict[str, str] = {}
        self._disposed = False

        # A context menu needs a QWidget parent to be owned; this QObject has
        # none, so the menu is kept alive here and released in dispose().
        self.menu = QMenu()
        self.menu.setObjectName("eyeTrackerTrayMenu")
        self._build_menu()

        self.tray = QSystemTrayIcon(self)
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_activated)

        self._connect_controller()
        hints = QGuiApplication.styleHints()
        scheme_changed = getattr(hints, "colorSchemeChanged", None)
        if scheme_changed is not None:
            scheme_changed.connect(self._on_theme_changed)

        self._sync_hotkeys()
        self._refresh_autostart()
        self._sync_state()

    # ------------------------------------------------------------------ public API
    @property
    def available(self) -> bool:
        """Whether the desktop has a system tray to show the icon in."""
        return QSystemTrayIcon.isSystemTrayAvailable()

    @property
    def state(self) -> TrackingState:
        """The tracking state the tray currently shows."""
        return self._state

    def show(self) -> None:
        """Show the icon (logs a hint when the desktop has no tray)."""
        if self._disposed:
            return
        self._refresh_icon(force=True)
        self.tray.show()
        if not self.available:
            log.warning(
                "No system tray is available; control the app with 'eye-tracker ctl …' "
                "or enable a tray/AppIndicator extension"
            )

    def hide(self) -> None:
        """Hide the icon (it can be shown again)."""
        if not self._disposed:
            self.tray.hide()

    def dispose(self) -> None:
        """Hide the icon and release the menu (call before quitting). Idempotent.

        Controller signals that still arrive afterwards are ignored.
        """
        if self._disposed:
            return
        self._disposed = True
        self.tray.hide()
        self.menu.deleteLater()

    def set_state(self, state: TrackingState) -> None:
        """Reflect a new tracking state in the icon, tooltip and menu."""
        if self._disposed:
            return
        if not isinstance(state, TrackingState):
            log.debug("Ignoring non-TrackingState %r", state)
            return
        self._state = state
        self._sync_state()

    def update_stats(self, stats: Mapping[str, Any]) -> None:
        """Take the controller's stats dict (``fps``, ``cpu_percent``, ``last_error``…)."""
        if self._disposed or not isinstance(stats, Mapping):
            return
        self._stats = dict(stats)
        self._sync_texts()
        # Stats arrive every few seconds: a cheap moment to notice that the
        # taskbar switched between light and dark (Qt only reports app themes).
        self._refresh_icon()

    def set_settings(self, settings: Settings) -> None:
        """Reflect changed settings (hotkey labels, gaze-dot check, notifications)."""
        if self._disposed or not isinstance(settings, Settings):
            return
        self._settings = settings
        self._sync_hotkeys()
        self.action_overlay.setChecked(settings.ui.show_gaze_overlay)

    def tooltip_text(self) -> str:
        """E.g. ``"Eye Tracker — Tracking · 4 fps · CPU 0.6 %"``."""
        parts = [f"{APP_NAME} — {self._state.label}"]
        fps = _number(self._stats.get("fps"))
        state = self._state
        if fps is not None and state.camera_active and state is not TrackingState.CAMERA_ERROR:
            parts.append(util.format_fps(fps))
        cpu = _number(self._stats.get("cpu_percent"))
        if cpu is not None:
            parts.append(util.format_cpu(cpu))
        return util.elide(" · ".join(parts), _TOOLTIP_LIMIT)

    def status_text(self) -> str:
        """The disabled first menu line, e.g. ``"Tracking"`` or the camera error."""
        state = self._state
        if state is TrackingState.CAMERA_ERROR:
            error = self._stats.get("last_error")
            if isinstance(error, str) and error.strip():
                return f"{state.label} — {util.elide(error, _STATUS_DETAIL_LIMIT)}"
        elif state is TrackingState.YIELDED:
            # "zoom.exe is running" / "Another app is using the camera".
            reason = getattr(self._controller, "yield_reason", "")
            if isinstance(reason, str) and reason.strip():
                return f"Paused — {util.elide(reason, _STATUS_DETAIL_LIMIT)}"
        return state.label

    def notify(
        self, title: str, message: str, *, force: bool = False, critical: bool = False
    ) -> bool:
        """Show a tray notification; returns whether one was shown.

        Honours ``settings.general.notifications`` and :meth:`mute_notifications`
        unless ``force`` (errors caused by a user action). ``critical`` shows a
        warning icon instead of the app icon.
        """
        return self._deliver(title, message, filtered=not force, critical=critical)

    def mute_notifications(self, seconds: float) -> None:
        """Suppress notifications for ``seconds`` (a start at login stays quiet).

        Only :meth:`notify` calls with ``force=True`` get through meanwhile.
        """
        self._muted_until = self._clock() + max(0.0, float(seconds))

    # ------------------------------------------------------------------ building
    def _add_action(
        self, text: str, slot: Callable[..., None], *, checkable: bool = False
    ) -> QAction:
        action = self.menu.addAction(text)
        # Keep Qt's macOS text heuristics from moving "About…"/"Quit" out of the
        # tray menu into an application menu this app does not have.
        action.setMenuRole(QAction.MenuRole.NoRole)
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        return action

    def _build_menu(self) -> None:
        menu = self.menu
        self.action_status = menu.addAction("")
        self.action_status.setEnabled(False)
        self.action_status.setMenuRole(QAction.MenuRole.NoRole)
        menu.addSeparator()
        self.action_pause = self._add_action("Pause tracking", self._on_pause)
        self.action_privacy = self._add_action("Privacy mode", self._on_privacy, checkable=True)
        menu.addSeparator()
        self.action_calibrate = self._add_action("Calibrate…", self._on_calibrate)
        self.action_overlay = self._add_action("Show gaze dot", self._on_overlay, checkable=True)
        self.action_overlay.setToolTip("Draw a dot where the tracker thinks you are looking")
        self.action_preview = self._add_action("Camera preview…", self._on_preview)
        self.action_preview.setToolTip("See what the camera sees (never saved)")
        menu.addSeparator()
        self.action_settings = self._add_action("Settings…", self._on_settings)
        self.action_autostart = self._add_action(
            "Start at login", self._on_autostart, checkable=True
        )
        menu.addSeparator()
        self.action_about = self._add_action(f"About {APP_NAME}", self._on_about)
        self.action_quit = self._add_action(f"Quit {APP_NAME}", self._on_quit)
        menu.setToolTipsVisible(True)
        menu.aboutToShow.connect(self._on_menu_about_to_show)

    def _connect_controller(self) -> None:
        for name, slot in (
            ("state_changed", self.set_state),
            ("stats_changed", self.update_stats),
            ("settings_changed", self.set_settings),
            ("notify", self._on_controller_notify),
        ):
            signal = getattr(self._controller, name, None)
            if signal is None:
                log.debug("Controller has no %s signal", name)
                continue
            signal.connect(slot)

    # ------------------------------------------------------------------- syncing
    def _flag(self, name: str, fallback: bool) -> bool:
        """A boolean controller property (``paused``, ``privacy``), else ``fallback``.

        The flags matter because the state shows only the strongest cause: a
        paused tracker in privacy mode reports ``PRIVACY``, yet "Resume" is right.
        """
        value = getattr(self._controller, name, None)
        return value if isinstance(value, bool) else fallback

    def _sync_state(self) -> None:
        state = self._state
        busy = state is TrackingState.CALIBRATING
        paused = self._flag("paused", state is TrackingState.PAUSED)
        self.action_pause.setText("Resume tracking" if paused else "Pause tracking")
        self.action_pause.setEnabled(not busy)
        self.action_privacy.setChecked(self._flag("privacy", state is TrackingState.PRIVACY))
        self.action_privacy.setEnabled(not busy)

        needs = state is TrackingState.NEEDS_CALIBRATION
        self.action_calibrate.setText("Calibrate now…" if needs else "Calibrate…")
        font = self.action_calibrate.font()
        font.setBold(needs)
        self.action_calibrate.setFont(font)
        self.action_calibrate.setEnabled(not busy)
        self.action_overlay.setChecked(self._settings.ui.show_gaze_overlay)
        self.action_status.setIcon(icons.status_dot_icon(_STATUS_COLORS.get(state, _MUTED_GREY)))
        self._sync_texts()
        self._sync_tooltips()
        self._refresh_icon()

    def _sync_texts(self) -> None:
        status = self.status_text()
        if self.action_status.text() != status:
            self.action_status.setText(status)
        tooltip = self.tooltip_text()
        # Each change is a round trip to the shell; skip unchanged tooltips.
        if self.tray.toolTip() != tooltip:
            self.tray.setToolTip(tooltip)

    def _hotkeys(self) -> dict[str, Hotkey]:
        """Global hotkeys by controller action (``toggle_tracking``, …) to show.

        Once the controller has a hotkey manager, only the combinations the OS
        accepted are shown: none on Wayland, none that another app owns. Before
        that (the manager is created by ``controller.start()``), the settings.
        """
        manager = getattr(self._controller, "hotkey_manager", None)
        registered = getattr(manager, "registered", None) if manager is not None else None
        if isinstance(registered, Mapping):
            return {
                str(name): hotkey
                for name, hotkey in registered.items()
                if isinstance(hotkey, Hotkey)
            }
        settings = self._settings.hotkeys
        if not settings.enabled:
            return {}
        hotkeys: dict[str, Hotkey] = {}
        for name in _HOTKEY_ACTIONS:
            hotkey = util.parse_hotkey_text(getattr(settings, name, ""))
            if hotkey is not None:
                hotkeys[name] = hotkey
        return hotkeys

    def _hotkey_actions(self) -> tuple[tuple[str, QAction], ...]:
        return (
            ("toggle_tracking", self.action_pause),
            ("toggle_privacy", self.action_privacy),
            ("recalibrate", self.action_calibrate),
        )

    def _sync_hotkeys(self) -> None:
        hotkeys = self._hotkeys()
        self._hotkey_labels = {name: format_hotkey(hk) for name, hk in hotkeys.items()}
        for name, action in self._hotkey_actions():
            hotkey = hotkeys.get(name)
            sequence = util.hotkey_sequence(hotkey) if hotkey is not None else None
            # The shortcut is only *displayed* next to the item (natively: a column
            # in Qt menus, a key equivalent on macOS, the DBusMenu "shortcut" on
            # Linux). WidgetShortcut limits it to the open menu; the global hotkey
            # itself is registered by the controller.
            action.setShortcut(sequence if sequence is not None else QKeySequence())
            action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
            action.setShortcutVisibleInContextMenu(True)
        self._sync_tooltips()

    def _sync_tooltips(self) -> None:
        state = self._state
        paused = self._flag("paused", state is TrackingState.PAUSED)
        calibrate_tip = _CALIBRATE_TIP
        reason = getattr(self._controller, "calibration_reason", "")
        if state is TrackingState.NEEDS_CALIBRATION and isinstance(reason, str) and reason.strip():
            calibrate_tip = f"Switching is off until you calibrate ({reason.strip()})"
        tips = {
            "toggle_tracking": _RESUME_TIP if paused else _PAUSE_TIP,
            "toggle_privacy": _PRIVACY_TIP,
            "recalibrate": calibrate_tip,
        }
        for name, action in self._hotkey_actions():
            label = self._hotkey_labels.get(name)
            # format_hotkey gives the platform's wording ("Ctrl+Alt+T", "⌃⌥T").
            action.setToolTip(f"{tips[name]} · {label}" if label else tips[name])

    def _refresh_icon(self, *, force: bool = False) -> None:
        if self._disposed:
            return
        mask = sys.platform == "darwin"
        dark = True if mask else util.system_tray_is_dark()
        key = (self._state, dark, mask)
        if key == self._icon_key and not force:
            return
        self._icon_key = key
        self.tray.setIcon(icons.tray_icon(self._state, dark=dark, mask=mask))

    def _refresh_autostart(self) -> None:
        backend = self._autostart
        try:
            supported = bool(getattr(backend, "is_supported", lambda: True)())
            enabled = supported and bool(backend.is_enabled())
        except Exception:
            log.debug("Reading the start-at-login state failed", exc_info=True)
            supported, enabled = True, False
        self.action_autostart.setVisible(supported)
        self.action_autostart.setChecked(enabled)

    # ------------------------------------------------------------------- helpers
    def _call(self, method: str, *args: Any) -> bool:
        """Call a controller method; log (never raise) on failure."""
        func = getattr(self._controller, method, None)
        if not callable(func):
            log.warning("Controller has no %s()", method)
            return False
        try:
            func(*args)
        except Exception:
            log.exception("controller.%s failed", method)
            return False
        return True

    def _deliver(
        self,
        title: str,
        message: str,
        *,
        filtered: bool,
        critical: bool = False,
        check_settings: bool = True,
    ) -> bool:
        """Show a message unless ``filtered`` and muted (or off in the settings)."""
        if self._disposed:
            return False
        if filtered and check_settings and not self._settings.general.notifications:
            log.debug("Notification off in settings: %s", title)
            return False
        if filtered and self._clock() < self._muted_until:
            log.debug("Notification muted: %s", title)
            return False
        if not self._messages_available():
            log.info("Notification (no tray to show it): %s — %s", title, message)
            return False
        self._show_message(title, message, critical)
        return True

    def _messages_available(self) -> bool:
        return self.tray.isVisible() and QSystemTrayIcon.supportsMessages()

    def _show_message(self, title: str, message: str, critical: bool) -> None:
        if critical:
            self.tray.showMessage(
                title, message, QSystemTrayIcon.MessageIcon.Warning, self.NOTIFICATION_MS
            )
        else:
            self.tray.showMessage(title, message, icons.app_icon(), self.NOTIFICATION_MS)

    # --------------------------------------------------------------------- slots
    def _on_pause(self) -> None:
        self._call("toggle_pause")
        # The controller normally answers with state_changed; if the state did not
        # change (e.g. paused while in privacy mode) the flags still did.
        self._sync_state()

    def _on_privacy(self, checked: bool) -> None:
        self._call("set_privacy", bool(checked))
        # The check mark always shows the real state, whatever the call did.
        self._sync_state()

    def _on_overlay(self, checked: bool) -> None:
        # Start from the controller's settings, not a cached copy, so that no other
        # change is undone by this one.
        updated = util.controller_settings(self._controller).copy()
        updated.ui.show_gaze_overlay = bool(checked)
        if self._call("apply_settings", updated):
            self._settings = util.controller_settings(self._controller)
        self.action_overlay.setChecked(self._settings.ui.show_gaze_overlay)

    def _on_autostart(self, checked: bool) -> None:
        try:
            if checked:
                self._autostart.enable(background=True)
            else:
                self._autostart.disable()
        except OSError as exc:  # AutostartError: the message is meant for the user
            log.warning("Could not change start at login: %s", exc)
            self.notify(
                "Start at login",
                str(exc) or "The login item could not be changed.",
                force=True,
                critical=True,
            )
        except Exception:
            log.exception("Changing start at login failed")
            self.notify(
                "Start at login",
                "The login item could not be changed; see the log for details.",
                force=True,
                critical=True,
            )
        self._refresh_autostart()

    def _on_calibrate(self) -> None:
        self.open_calibration.emit()

    def _on_preview(self) -> None:
        self.open_preview.emit()

    def _on_settings(self) -> None:
        self.open_settings.emit()

    def _on_about(self) -> None:
        self.open_about.emit()

    def _on_quit(self) -> None:
        self.quit_requested.emit()

    def _on_menu_about_to_show(self) -> None:
        if self._disposed:
            return
        # The login item can be changed elsewhere (Task Manager, System Settings),
        # and hotkeys are registered only once the controller has started.
        self._refresh_autostart()
        self._sync_hotkeys()
        self._sync_state()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if self._disposed:
            return
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.open_settings.emit()
        elif reason == QSystemTrayIcon.ActivationReason.MiddleClick:
            self._on_pause()

    def _on_controller_notify(self, title: str, message: str) -> None:
        # The controller has already applied settings.general.notifications: what
        # it sends while they are off is something the user asked to be told
        # (walk-away "notify", shoulder guard "notify") or a failed lock. Filtering
        # again would swallow exactly those. The startup mute still applies.
        self._deliver(str(title), str(message), filtered=True, check_settings=False)

    def _on_theme_changed(self, *_args: object) -> None:
        self._refresh_icon()
