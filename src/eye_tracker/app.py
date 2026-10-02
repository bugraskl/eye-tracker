"""The tray application: startup sequence, single instance and window management.

:func:`run_app` is what ``eye-tracker`` (``run``) executes. It is split into
:func:`build_app`, which creates everything, and :meth:`AppContext.exec`, which
runs the Qt event loop, so tests can build the whole app without entering the
loop. The startup order matters:

1. logging (console only for now), so native libraries are quietened before
   they load;
2. ``PlatformServices.prepare_process`` before the ``QApplication`` exists
   (DPI awareness and Qt environment variables), then the ``QApplication``;
3. the single-instance check: whoever takes the :class:`~eye_tracker.ipc.InstanceLock`
   is the instance. Otherwise (or if an older version answers) the running app
   is asked to show itself (or to calibrate; a login start only checks that it
   answers) and this process exits without touching the log file or the
   settings. An instance that is shutting down no longer answers but still
   holds the lock for a moment: the launch waits for it to exit and then
   becomes the instance;
4. file logging, settings, the command socket, the
   :class:`~eye_tracker.engine.controller.Controller` and the UI; once the
   event loop runs, the login item is pointed at this copy if it moved.

A process started at login (``--background``) stays quiet for
:data:`STARTUP_QUIET_S`. If the first-run setup was never finished, it then
offers the setup assistant with a notification, and a second launch (or
``ctl show``) opens the assistant.

:class:`EyeTrackerApp` owns the controller and every window. The windows talk to
the controller themselves (the tray, the gaze overlay, the countdown toast and
the privacy curtain follow its signals on their own); this class opens windows
on request (tray menu, IPC, hotkeys) and keeps at most one of each.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QEventLoop, QObject, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from . import APP_AUTHOR, APP_NAME, APP_SLUG, __version__, ipc, paths
from .cli import cli_command_text
from .config import Settings
from .logging_setup import install_qt_message_handler, set_level, setup_logging
from .platform import get_platform
from .platform.base import PlatformServices
from .types import TrackingState

if TYPE_CHECKING:
    from .engine.controller import Controller
    from .ui.about import AboutDialog
    from .ui.calibration_window import CalibrationWindow
    from .ui.countdown import CountdownToast
    from .ui.curtain import PrivacyCurtain
    from .ui.overlay import GazeOverlay
    from .ui.preview import PreviewWindow
    from .ui.settings_dialog import SettingsDialog
    from .ui.tray import TrayIcon
    from .ui.update_dialog import UpdateDialog
    from .ui.wizard import FirstRunWizard
    from .update.service import UpdateService

    ControllerFactory = Callable[[Settings, PlatformServices], Controller]

__all__ = ["AppContext", "AppOptions", "EyeTrackerApp", "build_app", "run_app"]

log = logging.getLogger(__name__)

#: ``calibration_required`` reasons that are explicit user requests (the
#: recalibrate hotkey, ``eye-tracker ctl calibrate``): the calibration opens at
#: once. Any other reason (a changed monitor layout, another vision backend) only
#: produces a notification that opens it when clicked: covering every screen
#: unannounced could hide a dialog the user is working with, such as the
#: operating system's "keep these display settings?" countdown.
EXPLICIT_CALIBRATION_REASONS = frozenset({"hotkey", "ipc", "user"})
#: Calibrations the app offered rather than the user asked for (``open_calibration``
#: reasons): closing one without saving brings the "Calibration needed" notice.
_OFFERED_CALIBRATIONS = frozenset({"wizard", "cli"})
#: With ``--background`` (login start) notifications stay quiet this long.
STARTUP_QUIET_S = 20.0
#: Clicking a "calibration needed" or "finish setup" notification later than
#: this does nothing.
PROMPT_CLICK_WINDOW_S = 600.0
#: Kinds of notifications that do something when clicked: open the calibration,
#: the setup assistant, the OS privacy settings of a missing permission, or the
#: update window.
PROMPT_CALIBRATE = "calibrate"
PROMPT_SETUP = "setup"
PROMPT_PERMISSION = "permission"
PROMPT_UPDATE = "update"
#: A ``permission_needed`` belongs to the controller notification emitted this
#: recently (the controller emits both back to back).
_PERMISSION_NOTICE_S = 1.0
#: States the user is absent in. Entering one forgets which calibration reasons
#: were announced, so the controller's reminder on their return is shown.
_ABSENT_STATES = frozenset({TrackingState.AWAY, TrackingState.LOCKED, TrackingState.PRIVACY})
#: How often Python gets control during the event loop, so Ctrl+C is handled.
SIGNAL_POLL_MS = 500
#: Delay before quitting on an IPC ``quit``, so the reply reaches the client.
_QUIT_DELAY_MS = 100
#: Margin after the startup mute ends before a deferred notification is shown.
_AFTER_QUIET_MS = 250
#: Settings that ``--camera`` / ``--backend`` override for one session.
_OVERRIDABLE = {"camera": "camera.device", "backend": "general.backend"}
#: Options that only a new instance applies: (``argparse`` attribute, option).
_STARTUP_ONLY_OPTIONS = (
    ("trace", "--trace"),
    ("camera", "--camera"),
    ("backend", "--backend"),
)
#: How long a launch keeps waiting for an instance that holds the lock but does
#: not answer, after :data:`~eye_tracker.ipc.STARTUP_WAIT_MS`: one that is shutting
#: down (its event loop has stopped while it releases the camera, which can take
#: a few seconds) exits within this time, and the launch then takes its place.
HANDOVER_WAIT_MS = 10_000
#: Reply timeout for each retry during :data:`HANDOVER_WAIT_MS`.
_HANDOVER_REPLY_MS = 500
#: Pause between two retries during :data:`HANDOVER_WAIT_MS`.
_HANDOVER_RETRY_MS = 250
#: Part of the reply of an instance that cannot take commands yet or any more.
_BUSY_MARKER = "is starting or shutting down"


@dataclass(frozen=True, slots=True)
class AppOptions:
    """Command-line switches that change how the app starts."""

    #: Started at login: no first-run wizard, no startup notifications, and an
    #: instance that is already running is not asked to show itself.
    background: bool = False
    #: Open the calibration right after start.
    calibrate: bool = False
    #: ``--log-level`` was given, so the settings' level is ignored.
    log_level_locked: bool = False


# ============================================================================ run
def run_app(args: argparse.Namespace) -> int:
    """Start the tray app and run the Qt event loop. Returns the exit code.

    Honours ``args.background``, ``args.calibrate``, ``args.camera``,
    ``args.backend``, ``args.config_dir`` and ``args.log_level``; missing
    attributes mean "not given".
    """
    return build_app(args).exec()


@dataclass(slots=True)
class AppContext:
    """What :func:`build_app` created. :meth:`exec` runs it.

    ``exit_code`` is set when there is nothing to run: the request was handed to
    an instance that is already running, or startup failed.
    """

    qt_app: QApplication
    app: EyeTrackerApp | None = None
    server: ipc.InstanceServer | None = None
    exit_code: int | None = None

    @property
    def controller(self) -> Controller | None:
        return self.app.controller if self.app is not None else None

    def exec(self) -> int:
        """Run the event loop until the app quits, then shut down. Returns the exit code."""
        if self.app is None or self.exit_code is not None:
            return self.exit_code if self.exit_code is not None else 0
        with _quit_on_signals(self.app.quit):
            code = int(self.qt_app.exec())
        self.shutdown()
        log.info("%s exited (code %s)", APP_NAME, code)
        return code

    def shutdown(self) -> None:
        """Stop accepting commands, close every window and release the camera. Idempotent."""
        if self.server is not None:
            self.server.close()
        if self.app is not None:
            self.app.shutdown()


def build_app(
    args: argparse.Namespace,
    *,
    platform: PlatformServices | None = None,
    controller_factory: ControllerFactory | None = None,
    controller_kwargs: Mapping[str, Any] | None = None,
) -> AppContext:
    """Create the ``QApplication`` (unless one exists), check for a running
    instance, then create and start the controller, the command socket and the UI.

    Args:
        args: Parsed command line (see :func:`run_app`).
        platform: OS integration; defaults to :func:`~eye_tracker.platform.get_platform`.
        controller_factory: ``(settings, platform) -> Controller``; defaults to
            :class:`~eye_tracker.engine.controller.Controller`.
        controller_kwargs: Extra keyword arguments for the default controller
            (its test seams: ``worker_factory``, ``monitors_provider``, ``cursor``,
            ``hotkey_manager_factory``, ``clock``).
    """
    options = AppOptions(
        background=bool(getattr(args, "background", False)),
        calibrate=bool(getattr(args, "calibrate", False)),
        log_level_locked=bool(getattr(args, "log_level", None)),
    )
    config_dir = getattr(args, "config_dir", None)
    if config_dir:
        paths.set_base_override(Path(config_dir))
    level = getattr(args, "log_level", None) or "INFO"
    console = _isatty(sys.stderr)
    # The log file belongs to the running instance; a process that only hands a
    # request over to it must not write (or rotate) it, so file logging starts
    # after the single-instance check.
    setup_logging(level, console=console, log_to_file=False)

    services = platform if platform is not None else get_platform()
    existing = QApplication.instance()
    if existing is None:
        # Must happen before the QApplication exists (DPI awareness, Qt env vars).
        services.prepare_process()
        qt_app = QApplication([sys.argv[0] if sys.argv and sys.argv[0] else APP_SLUG])
    elif isinstance(existing, QApplication):
        qt_app = existing
    else:
        raise RuntimeError("A non-GUI Qt application already exists; cannot start the tray app")
    _configure_qt_app(qt_app)
    # macOS: no Dock icon or application menu for a tray app. The .app bundle's
    # LSUIElement does this too; this covers running from source.
    accessory = getattr(services, "set_accessory_app", None)
    if callable(accessory):
        try:
            accessory()
        except Exception:
            log.debug("set_accessory_app failed", exc_info=True)

    # Single instance. Taking the lock is atomic, so of two launches racing each
    # other exactly one becomes the instance; the other hands its request over
    # (waiting for the winner to start listening) and leaves.
    request = _handover_request(options)
    lock = ipc.InstanceLock()
    owner = lock.acquire()
    try:
        # The owner asks too: an instance of a version without the lock answers.
        # A launch that lost the lock waits for the winner to start listening.
        reply = ipc.send_command(request, wait_ms=0 if owner else ipc.STARTUP_WAIT_MS)
        if not owner and (reply is None or _is_busy(reply)):
            owner, reply = _wait_for_handover(lock, request)
    except BaseException:
        lock.release()  # never keep a later launch out after failing here
        raise
    if reply is not None and not _is_busy(reply):
        lock.release()
        if reply.startswith("error"):
            log.warning("The running instance refused %r: %s", request, reply)
            return AppContext(qt_app, exit_code=1)
        log.info("%s is already running; asked it to %s", APP_NAME, request)
        _warn_startup_options_ignored(args)
        return AppContext(qt_app, exit_code=0)
    if not owner or reply is not None:
        # Nobody answered in time, or only "starting or shutting down" (from an
        # older version without the lock, when this process holds it).
        lock.release()
        log.error("Another instance (lock %s) does not answer; exiting", lock.path)
        _show_error(qt_app, f"{APP_NAME} is already running but does not respond.")
        return AppContext(qt_app, exit_code=1)

    app: EyeTrackerApp | None = None
    try:
        app = _create_app(
            qt_app,
            args,
            services,
            options,
            level=level,
            console=console,
            controller_factory=controller_factory,
            controller_kwargs=controller_kwargs,
        )
        # From here on the server owns the lock and releases it in close().
        server = ipc.InstanceServer(app.handle_command, app, lock=lock)
    except BaseException:
        lock.release()  # never keep a later launch out after failing here
        if app is not None:
            app.shutdown()
        raise
    if not server.listen():
        if server.another_instance_running:
            # Only possible where files cannot be locked: another instance
            # claimed the socket after our check, or it hangs.
            log.error("Another instance is running but does not answer; exiting")
            _show_error(qt_app, f"{APP_NAME} is already running but does not respond.")
            server.close()
            app.shutdown()
            return AppContext(qt_app, exit_code=1)
        log.warning("Command socket unavailable; '%s' will not work", cli_command_text("ctl"))

    try:
        app.start()
    except Exception as exc:
        log.exception("%s could not start", APP_NAME)
        server.close()
        app.shutdown()
        _show_error(
            qt_app,
            f"{APP_NAME} could not start:\n\n{exc}\n\n"
            f"Details are in the log file:\n{paths.log_file()}",
        )
        return AppContext(qt_app, exit_code=1)
    return AppContext(qt_app, app, server)


def _create_app(
    qt_app: QApplication,
    args: argparse.Namespace,
    services: PlatformServices,
    options: AppOptions,
    *,
    level: str,
    console: bool,
    controller_factory: ControllerFactory | None,
    controller_kwargs: Mapping[str, Any] | None,
) -> EyeTrackerApp:
    """File logging, settings (with command-line overrides) and the app object.

    Runs once this process is known to be the instance, so it may write the log
    file and read the settings.
    """
    setup_logging(level, console=console)
    install_qt_message_handler()
    settings = Settings.load(paths.settings_file())
    if not options.log_level_locked:
        set_level(settings.general.log_level)
    overrides = _apply_overrides(settings, args)
    log.info(
        "%s %s starting (Python %s, %s)",
        APP_NAME,
        __version__,
        sys.version.split()[0],
        sys.platform,
    )

    trace = getattr(args, "trace", None)
    if trace:
        controller_kwargs = {**dict(controller_kwargs or {}), "trace_path": trace}
    if controller_factory is None and controller_kwargs:
        controller_factory = _controller_factory(dict(controller_kwargs))
    return EyeTrackerApp(
        qt_app,
        settings,
        services,
        options,
        controller_factory=controller_factory,
        session_overrides=overrides,
    )


def _controller_factory(kwargs: dict[str, Any]) -> ControllerFactory:
    def factory(settings: Settings, services: PlatformServices) -> Controller:
        from .engine.controller import Controller

        return Controller(settings, services, **kwargs)

    return factory


# ========================================================================= the app
class EyeTrackerApp(QObject):
    """Owns the controller, the tray icon and the windows of the running app.

    Args:
        qt_app: The ``QApplication``.
        settings: Settings to start with (command-line overrides applied).
        services: OS integration.
        options: Startup switches.
        controller_factory: ``(settings, services) -> Controller``; defaults to
            :class:`~eye_tracker.engine.controller.Controller`.
        session_overrides: ``{"section.field": (saved value, session value)}`` for
            settings overridden on the command line. The session value stays out
            of the settings file when other settings are saved.
        clock: Monotonic time source (notification click window).
    """

    def __init__(
        self,
        qt_app: QApplication,
        settings: Settings,
        services: PlatformServices,
        options: AppOptions | None = None,
        *,
        controller_factory: ControllerFactory | None = None,
        session_overrides: Mapping[str, tuple[str, str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._qt_app = qt_app
        self._settings = settings
        self._services = services
        self._options = options or AppOptions()
        self._controller_factory = controller_factory
        self._overrides = dict(session_overrides or {})
        self._clock = clock
        self._controller: Controller | None = None
        self._started = False
        self._closed = False
        self._quit_hooked = False

        self._tray: TrayIcon | None = None
        self._overlay: GazeOverlay | None = None
        self._countdown: CountdownToast | None = None
        self._curtain: PrivacyCurtain | None = None
        self._settings_dialog: SettingsDialog | None = None
        self._calibration: CalibrationWindow | None = None
        self._preview: PreviewWindow | None = None
        self._about: AboutDialog | None = None
        self._wizard: FirstRunWizard | None = None
        self._updates: UpdateService | None = None
        self._update_dialog: UpdateDialog | None = None
        #: Reasons announced with a "calibration needed" notification since the
        #: calibration was last usable (cleared when it is usable again).
        self._announced: set[str] = set()
        #: The clickable notification on screen: (``PROMPT_*`` kind, when shown).
        self._prompt: tuple[str, float] | None = None
        #: The permission a ``PROMPT_PERMISSION`` notification is about.
        self._prompt_permission: str | None = None
        #: When the controller's latest notification was shown (see
        #: _on_permission_needed); ``None`` once used or when it was not shown.
        self._last_notice_at: float | None = None
        #: The controller's latest notification, shown or not: (title, message, when).
        self._last_notice: tuple[str, str, float] | None = None
        #: A reason that could not be announced while notifications were muted.
        self._deferred_reason: str | None = None
        #: A permission notice swallowed by the startup mute: (name, title, message).
        self._deferred_permission: tuple[str, str, str] | None = None

    # ---------------------------------------------------------------- accessors
    @property
    def controller(self) -> Controller | None:
        return self._controller

    @property
    def tray(self) -> TrayIcon | None:
        return self._tray

    @property
    def overlay(self) -> GazeOverlay | None:
        return self._overlay

    @property
    def countdown(self) -> CountdownToast | None:
        return self._countdown

    @property
    def curtain(self) -> PrivacyCurtain | None:
        return self._curtain

    @property
    def settings_dialog(self) -> SettingsDialog | None:
        return self._settings_dialog

    @property
    def calibration_window(self) -> CalibrationWindow | None:
        return self._calibration

    @property
    def preview(self) -> PreviewWindow | None:
        return self._preview

    @property
    def wizard(self) -> FirstRunWizard | None:
        return self._wizard

    @property
    def closed(self) -> bool:
        return self._closed

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Create the controller and the UI, start tracking and open startup windows."""
        if self._started or self._closed:
            return
        self._started = True
        controller = self._make_controller()
        self._controller = controller
        controller.calibration_required.connect(self._on_calibration_required)
        controller.settings_changed.connect(self._on_settings_changed)
        controller.ui_requested.connect(self._on_ui_requested)
        controller.state_changed.connect(self._on_state_changed)

        from .ui.countdown import CountdownToast
        from .ui.curtain import PrivacyCurtain
        from .ui.overlay import GazeOverlay
        from .ui.tray import TrayIcon

        # These follow the controller's signals by themselves: the toast
        # away_warning/away_cancelled, the curtain guard_changed, the overlay
        # gaze_changed and settings.ui.show_gaze_overlay.
        tray = TrayIcon(controller, self)
        self._tray = tray
        self._overlay = GazeOverlay(controller, self)
        self._countdown = CountdownToast(controller)
        curtain = PrivacyCurtain(controller, self)
        self._curtain = curtain
        # Connected after the tray: any other notification replaces ours.
        controller.notify.connect(self._on_controller_notify)

        tray.open_settings.connect(self.open_settings)
        tray.open_calibration.connect(self._on_tray_calibrate)
        tray.open_preview.connect(self.open_preview)
        tray.open_about.connect(self.open_about)
        tray.open_update.connect(self.open_update)
        tray.quit_requested.connect(self.quit)
        tray.tray.messageClicked.connect(self._on_message_clicked)
        # Esc or the button on the curtain: switching resumes (the guard stays on).
        curtain.dismissed.connect(controller.dismiss_curtain)
        controller.permission_needed.connect(self._on_permission_needed)
        if self._options.background:
            tray.mute_notifications(STARTUP_QUIET_S)
        tray.show()
        self._qt_app.aboutToQuit.connect(self.shutdown)
        self._quit_hooked = True

        controller.start()
        self._start_updates(tray)
        # Windows open once the event loop runs, so the tray is already in place.
        QTimer.singleShot(0, self, self._after_start)

    def shutdown(self) -> None:
        """Close every window, then stop tracking (releases the camera). Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._quit_hooked:
            self._quit_hooked = False
            with contextlib.suppress(RuntimeError, TypeError):
                self._qt_app.aboutToQuit.disconnect(self.shutdown)
        # Windows first, while the controller still runs: an open calibration
        # reports its cancellation, the preview turns the preview stream off.
        calibration, self._calibration = self._calibration, None
        if calibration is not None:
            with contextlib.suppress(Exception):
                calibration.cancel()
            calibration.deleteLater()
        if self._updates is not None:
            self._updates.shutdown()
        widgets: list[QWidget | None] = [
            self._wizard,
            self._settings_dialog,
            self._about,
            self._preview,
            self._countdown,
            self._update_dialog,
        ]
        self._wizard = self._settings_dialog = self._about = self._preview = None
        self._countdown = self._update_dialog = None
        for widget in widgets:
            if widget is not None:
                _dispose_widget(widget)
        for helper in (self._overlay, self._curtain):
            if helper is not None:
                with contextlib.suppress(Exception):
                    helper.close()
        if self._controller is not None:
            try:
                self._controller.shutdown()
            except Exception:
                log.exception("Controller shutdown failed")
        if self._tray is not None:
            with contextlib.suppress(Exception):
                self._tray.dispose()

    def quit(self) -> None:
        """Leave the event loop (shutdown follows via ``aboutToQuit``)."""
        log.info("Quit requested")
        self._qt_app.quit()

    # ----------------------------------------------------------------- commands
    def handle_command(self, command: str) -> str:
        """IPC entry point: :meth:`Controller.handle_command` executes it (UI
        commands come back through ``ui_requested``)."""
        controller = self._controller
        if controller is None or self._closed:
            # A second launch that gets this waits for the instance (_is_busy).
            return f"error: {APP_NAME} {_BUSY_MARKER}; try again"
        return str(controller.handle_command(command))

    def present(self) -> None:
        """Answer a second launch or ``ctl show``: bring up whatever needs attention.

        An open calibration, wizard or settings window is raised. If the first-run
        setup was never finished (e.g. the app was only ever started at login),
        the setup assistant opens. Otherwise the tray menu pops up next to the
        tray icon; without a system tray (e.g. GNOME without an AppIndicator
        extension) the settings window opens instead.
        """
        if self._closed:
            return
        if self._raise_calibration():
            return
        for window in (self._wizard, self._settings_dialog):
            if window is not None and window.isVisible():
                _present(window)
                return
        if self._setup_pending():
            self.open_wizard()
            return
        if not self._popup_tray_menu():
            self.open_settings()

    # ------------------------------------------------------------------ windows
    def open_settings(self) -> None:
        """Show the settings window (a fresh one unless it is already open)."""
        if self._closed or self._controller is None or self._raise_calibration():
            return
        if self._wizard is not None and self._wizard.isVisible():
            _present(self._wizard)
            return
        dialog = self._settings_dialog
        if dialog is None or not dialog.isVisible():
            from .ui.settings_dialog import SettingsDialog

            if dialog is not None:
                dialog.deleteLater()
            dialog = SettingsDialog(self._controller)
            dialog.calibration_requested.connect(self._on_settings_calibrate)
            dialog.setup_requested.connect(self._on_settings_setup)
            dialog.finished.connect(functools.partial(self._on_settings_closed, dialog))
            self._settings_dialog = dialog
        _present(dialog)

    def open_calibration(self, reason: str = "user") -> None:
        """Open the calibration on every monitor, or raise it if it is already open."""
        if self._closed or self._controller is None:
            return
        window = self._calibration
        if window is not None and window.is_active:
            window.start()  # raises the existing windows
            return
        from .ui.calibration_window import CalibrationWindow

        log.info("Opening the calibration (%s)", reason)
        self._prompt = None
        window = CalibrationWindow(self._controller, self)
        window.finished.connect(functools.partial(self._on_calibration_finished, window, reason))
        # Assigned before start(): start() emits finished(False) at once when
        # there is no monitor, and the slot clears this reference.
        self._calibration = window
        window.start()

    def open_preview(self) -> None:
        """Show the camera preview (one instance; closing only hides it)."""
        if self._closed or self._controller is None or self._raise_calibration():
            return
        if self._preview is None:
            from .ui.preview import PreviewWindow

            self._preview = PreviewWindow(self._controller)
        _present(self._preview)

    def open_about(self) -> None:
        """Show the About dialog."""
        if self._closed or self._raise_calibration():
            return
        if self._about is None or not self._about.isVisible():
            from .ui.about import AboutDialog

            if self._about is not None:
                self._about.deleteLater()
            self._about = AboutDialog()
        _present(self._about)

    def open_update(self) -> None:
        """Show the update window (it looks for an update unless one is known)."""
        updates = self._updates
        if self._closed or updates is None or self._raise_calibration():
            return
        if self._update_dialog is None:
            from .ui.update_dialog import UpdateDialog

            self._update_dialog = UpdateDialog(updates)
        self._update_dialog.present()

    def open_wizard(self) -> None:
        """Show the first-run wizard."""
        if self._closed or self._controller is None or self._raise_calibration():
            return
        if self._wizard is not None:
            _present(self._wizard)
            return
        from .ui.wizard import FirstRunWizard

        wizard = FirstRunWizard(self._controller)
        wizard.calibration_requested.connect(self._on_wizard_calibrate)
        wizard.finished.connect(functools.partial(self._on_wizard_finished, wizard))
        self._wizard = wizard
        _present(wizard)

    # -------------------------------------------------------------------- slots
    def _after_start(self) -> None:
        controller = self._controller
        if self._closed or controller is None:
            return
        self._refresh_autostart()
        self._remind_privacy_mode()
        setup_offered = False
        if self._setup_pending():
            if self._options.background:
                # A login start stays quiet; offer the setup once that is over.
                self._after_quiet_period(self._prompt_setup)
                setup_offered = True  # the setup offers the calibration itself
            else:
                try:
                    self.open_wizard()
                except Exception:
                    log.exception("The first-run wizard could not be opened")
                else:
                    return  # the wizard offers the calibration itself
        if self._options.calibrate:
            self.open_calibration("cli")
        elif not setup_offered:
            # A login start too: the notice waits for the end of its quiet period
            # (see _suggest_calibration). Skipping it there meant a setup that was
            # never calibrated and starts at login was never told why nothing
            # switches.
            self._suggest_calibration(controller.calibration_reason)

    def _on_calibration_required(self, reason: str) -> None:
        if reason in EXPLICIT_CALIBRATION_REASONS:
            self.open_calibration(reason)
        else:
            self._suggest_calibration(reason)

    def _on_tray_calibrate(self) -> None:
        self.open_calibration("tray")

    def _on_settings_calibrate(self) -> None:
        self.open_calibration("settings")

    def _on_settings_setup(self) -> None:
        # Opened right away rather than "at next start": waiting meant storing
        # first_run_done=False, which also switches the walk-away lock to a
        # notification until the assistant is finished (see the controller).
        self.open_wizard()

    def _on_wizard_calibrate(self) -> None:
        self.open_calibration("wizard")

    def _on_calibration_finished(self, window: CalibrationWindow, reason: str, saved: bool) -> None:
        if self._calibration is window:
            self._calibration = None
        window.deleteLater()
        log.info("Calibration %s", "saved" if saved else "closed without saving")
        controller = self._controller
        if not saved and reason in _OFFERED_CALIBRATIONS and controller is not None:
            # Offered rather than asked for (setup assistant, --calibrate): whoever
            # put it off learns what that means, and that a click brings it back.
            self._suggest_calibration(controller.calibration_reason)

    def _on_settings_closed(self, dialog: SettingsDialog, _result: int) -> None:
        if self._settings_dialog is dialog:
            self._settings_dialog = None
        dialog.deleteLater()

    def _on_wizard_finished(self, wizard: FirstRunWizard, _result: int) -> None:
        if self._wizard is wizard:
            self._wizard = None
        wizard.deleteLater()
        controller = self._controller
        if self._closed or controller is None or wizard.wants_calibration:
            return
        # Finished (or cancelled) without calibrating: say why switching is off.
        self._suggest_calibration(controller.calibration_reason)

    def _on_settings_changed(self, settings: object) -> None:
        if not isinstance(settings, Settings):
            return
        self._settings = settings
        if not self._options.log_level_locked:
            set_level(settings.general.log_level)
        if self._updates is not None:
            self._updates.set_auto(settings.updates.check)
        self._save_without_overrides(settings)

    def _on_ui_requested(self, command: str) -> None:
        if command == "show":
            self.present()
        elif command == "settings":
            self.open_settings()
        elif command == "quit":
            QTimer.singleShot(_QUIT_DELAY_MS, self, self.quit)
        else:
            log.debug("Ignoring unknown UI request %r", command)

    def _on_state_changed(self, state: object) -> None:
        controller = self._controller
        if not self._announced or controller is None:
            return
        if controller.is_calibrated or state in _ABSENT_STATES:
            # Usable again (recalibrated, or back at a known desk): the next time
            # it stops being usable deserves a notification, even for a reason
            # announced before (docking at a new desk is "the monitor layout
            # changed" every time). And after being away, locked or private,
            # the controller reminds the user of a calibration that is still
            # unusable, with the same reason as before: that must be shown too.
            self._announced.clear()

    def _on_controller_notify(self, title: str, message: str) -> None:
        # The tray shows one message at a time; ours has been replaced.
        self._prompt = None
        self._prompt_permission = None
        tray = self._tray
        now = self._clock()
        shown = tray is not None and tray.muted_for <= 0
        self._last_notice_at = now if shown else None
        self._last_notice = (str(title), str(message), now)

    def _on_permission_needed(self, name: str) -> None:
        """Make the notification that explains a missing permission clickable.

        The controller emits ``permission_needed`` right after that ``notify``
        (or without one when notifications are off, and then there is nothing to
        click). Clicking opens the OS privacy settings for ``name``.

        The controller says it only once, so a notice swallowed by the quiet
        period of a login start (typically the Accessibility check at startup)
        is shown once that period is over, if the permission is still missing.
        """
        shown_at, self._last_notice_at = self._last_notice_at, None
        notice, self._last_notice = self._last_notice, None
        now = self._clock()
        if shown_at is not None and now - shown_at <= _PERMISSION_NOTICE_S:
            self._prompt = (PROMPT_PERMISSION, now)
            self._prompt_permission = str(name)
            return
        tray = self._tray
        if (
            notice is None
            or now - notice[2] > _PERMISSION_NOTICE_S
            or tray is None
            or tray.muted_for <= 0
        ):
            return  # nothing was said (notifications off), so nothing to repeat
        if self._deferred_permission is None:
            self._after_quiet_period(self._announce_deferred_permission)
        # One slot: a click can only open the settings of the notice shown last.
        self._deferred_permission = (str(name), notice[0], notice[1])

    def _announce_deferred_permission(self) -> None:
        deferred, self._deferred_permission = self._deferred_permission, None
        tray = self._tray
        if deferred is None or self._closed or tray is None:
            return
        name, title, message = deferred
        if not self._permission_still_needed(name):
            return
        # The controller applied the notifications setting when it sent it.
        if tray.notify(title, message, force=True):
            self._prompt = (PROMPT_PERMISSION, self._clock())
            self._prompt_permission = name

    def _permission_still_needed(self, name: str) -> bool:
        """Whether permission ``name`` still blocks something (see _on_permission_needed)."""
        controller = self._controller
        if controller is None:
            return False
        if name == "camera":
            return controller.state is TrackingState.CAMERA_ERROR
        if name == "accessibility":
            try:
                return str(self._services.accessibility_status()) in ("missing", "stale")
            except Exception:
                log.debug("accessibility_status() failed", exc_info=True)
        return True

    def _on_message_clicked(self) -> None:
        prompt, self._prompt = self._prompt, None
        permission, self._prompt_permission = self._prompt_permission, None
        controller = self._controller
        if prompt is None or controller is None:
            return
        kind, shown_at = prompt
        if self._clock() - shown_at > PROMPT_CLICK_WINDOW_S:
            return
        if kind == PROMPT_SETUP and self._setup_pending():
            self.open_wizard()
        elif kind == PROMPT_CALIBRATE and not controller.is_calibrated:
            self.open_calibration("notification")
        elif kind == PROMPT_PERMISSION and permission:
            self._open_permission_settings(permission)
        elif kind == PROMPT_UPDATE:
            self.open_update()

    def _start_updates(self, tray: TrayIcon) -> None:
        """Create the update service (it looks for updates only if the user turned
        that on, or asks) and connect it to the tray."""
        from .update.service import UpdateService

        updates = UpdateService(self)
        self._updates = updates
        updates.state_changed.connect(self._on_update_state)
        updates.announce.connect(self._on_update_announced)
        updates.start(self._settings.updates.check)

    def _on_update_state(self, state: object) -> None:
        from .update.service import Phase, UpdateState

        tray = self._tray
        if tray is None or not isinstance(state, UpdateState):
            return
        release = state.release
        if state.phase is Phase.UP_TO_DATE:
            tray.set_update_available(None)
        elif release is not None and state.phase in (
            Phase.AVAILABLE,
            Phase.DOWNLOADING,
            Phase.INSTALLING,
            Phase.FAILED,
        ):
            tray.set_update_available(release.version)

    def _on_update_announced(self, release: object) -> None:
        """A newer version was found by the daily check: say so once."""
        tray = self._tray
        version = getattr(release, "version", None)
        if tray is None or not version:
            return
        shown = tray.notify(
            "Update available",
            f"{APP_NAME} {version} is available. Click here to update.",
        )
        if shown:
            self._prompt = (PROMPT_UPDATE, self._clock())

    # ------------------------------------------------------------------ helpers
    def _make_controller(self) -> Controller:
        if self._controller_factory is not None:
            return self._controller_factory(self._settings, self._services)
        from .engine.controller import Controller

        return Controller(self._settings, self._services)

    def _popup_tray_menu(self) -> bool:
        tray = self._tray
        if tray is None or not tray.available or not tray.tray.isVisible():
            return False
        geometry = tray.tray.geometry()
        # The icon may sit in an overflow area without a geometry.
        point = geometry.center() if geometry.isValid() and geometry.width() > 0 else None
        tray.menu.popup(point if point is not None else QCursor.pos())
        tray.menu.activateWindow()
        return True

    def _suggest_calibration(self, reason: str) -> None:
        """Tell the user that switching needs a calibration.

        Once per reason until the calibration is usable again. A notification
        that could not be shown does not count: one suppressed by the quiet
        period after a login start is shown when that period ends.
        """
        controller = self._controller
        tray = self._tray
        if controller is None or tray is None or controller.is_calibrated:
            return
        if self._calibration is not None and self._calibration.is_active:
            return
        if len(controller.monitors()) < 2 or reason in self._announced:
            return  # with one monitor there is nothing to switch between
        if not controller.settings.switching.enabled:
            return  # switching is off on purpose: a calibration changes nothing
        if getattr(controller, "privacy", False):
            # The camera is off on purpose; the controller reminds the user once
            # privacy mode ends (_remind_calibration).
            return
        detail = f" ({reason})" if reason else ""
        shown = tray.notify(
            "Calibration needed",
            f"Switching monitors is off until you calibrate{detail}. "
            "Click here or choose “Calibrate now…” in the tray menu.",
        )
        if shown:
            self._announced.add(reason)
            self._prompt = (PROMPT_CALIBRATE, self._clock())
        elif tray.muted_for > 0:
            # Swallowed by the quiet period of a login start: say it afterwards.
            if self._deferred_reason is None:
                self._after_quiet_period(self._announce_deferred)
            self._deferred_reason = reason

    def _announce_deferred(self) -> None:
        reason, self._deferred_reason = self._deferred_reason, None
        controller = self._controller
        if reason is not None and not self._closed and controller is not None:
            # The reason may be stale by now; the current one is what matters.
            self._suggest_calibration(controller.calibration_reason or reason)

    def _remind_privacy_mode(self) -> None:
        """Say that privacy mode is still on from the previous run.

        It is remembered across restarts so that the camera never comes back on
        by itself (``privacy.remember_privacy_mode``); this tells the user why
        nothing switches and walk-away detection is off. A notice swallowed by
        the quiet period of a login start is shown once that period is over.
        """
        controller, tray = self._controller, self._tray
        if self._closed or controller is None or tray is None:
            return
        if not getattr(controller, "privacy_restored", False):
            return
        shown = tray.notify(
            "Privacy mode is still on",
            "The camera stays off, as you left it, so monitor switching and walk-away "
            "detection are paused. Turn privacy mode off in the tray menu.",
        )
        if not shown and tray.muted_for > 0:
            self._after_quiet_period(self._remind_privacy_mode)

    def _prompt_setup(self) -> None:
        """After a quiet login start: offer the first-run setup that never happened."""
        tray = self._tray
        if self._closed or tray is None or not self._setup_pending():
            return
        if self._wizard is not None or (
            self._calibration is not None and self._calibration.is_active
        ):
            return
        shown = tray.notify(
            f"Finish setting up {APP_NAME}",
            "Check the camera and choose what happens when you walk away. "
            "Click here to open the setup assistant.",
        )
        if shown:
            self._prompt = (PROMPT_SETUP, self._clock())

    def _after_quiet_period(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` once the startup notification mute is over."""
        tray = self._tray
        delay = tray.muted_for if tray is not None else 0.0
        QTimer.singleShot(int(delay * 1000) + _AFTER_QUIET_MS, self, callback)

    def _setup_pending(self) -> bool:
        """The first-run setup was never finished.

        Nothing in the UI sets ``first_run_done`` back to ``False``: running the
        assistant again from the settings opens it at once (see
        :meth:`_on_settings_setup`), so the walk-away lock stays in force.
        """
        controller = self._controller
        return controller is not None and not controller.settings.general.first_run_done

    def _raise_calibration(self) -> bool:
        """Raise an open calibration instead of opening another window. Returns
        whether one was open.

        Every other window would open underneath its topmost full-screen
        surfaces and take the keyboard from it (Space, R and Esc would stop
        working with no visible reason).
        """
        calibration = self._calibration
        if calibration is None or not calibration.is_active:
            return False
        calibration.start()  # only raises and refocuses the existing surfaces
        return True

    def _open_permission_settings(self, name: str) -> None:
        """Open the OS privacy settings page for permission ``name``. Never raises."""
        try:
            opened = self._services.open_permission_settings(name)
        except Exception:
            log.warning("Could not open the %s privacy settings", name, exc_info=True)
            return
        if not opened:
            log.info("No privacy settings page to open for %s here", name)

    def _refresh_autostart(self) -> None:
        """Point the login item at this copy of the app if it moved (e.g. an
        update to another folder, a replaced AppImage). Never raises."""
        try:
            from .platform import autostart

            if autostart.is_supported():
                autostart.refresh()
        except Exception:
            log.warning("Could not check the start-at-login entry", exc_info=True)

    def _save_without_overrides(self, settings: Settings) -> None:
        """The controller saved ``settings``; keep command-line overrides out of the file."""
        restore = {
            key: saved
            for key, (saved, session) in self._overrides.items()
            if _get_setting(settings, key) == session and saved != session
        }
        if not restore:
            return
        on_disk = settings.copy()
        for key, value in restore.items():
            _set_setting(on_disk, key, value)
        try:
            on_disk.save(paths.settings_file())
        except OSError as exc:
            log.warning("Could not save the settings: %s", exc)


# ======================================================================= helpers
def _handover_request(options: AppOptions) -> str:
    """What a launch asks an instance that is already running to do.

    A login start (``--background``) must not pop anything up there (the tray
    menu, the setup assistant or the settings take the keyboard from whatever
    the user is typing in), so it only asks for ``status``, which just proves
    that the instance answers.
    """
    if options.calibrate:
        return "calibrate"
    return "status" if options.background else "show"


def _is_busy(reply: str) -> bool:
    """Whether ``reply`` says the instance cannot take commands yet or any more."""
    return reply.startswith("error") and _BUSY_MARKER in reply


def _wait_for_handover(lock: ipc.InstanceLock, request: str) -> tuple[bool, str | None]:
    """Wait for an instance that holds ``lock`` but did not answer ``request``.

    Such an instance is usually shutting down: ``aboutToQuit`` stops the camera
    with the event loop already stopped, while the lock (and the socket, which
    still accepts connections) are released only afterwards. Giving up at once
    would leave nothing running once it has exited: the classic "quit from the
    tray and start again right away". So until :data:`HANDOVER_WAIT_MS` has
    passed, this takes the lock as soon as it is free, or hands ``request``
    over if the instance answers after all.

    Returns ``(owner, reply)``: ``(True, None)`` when this process now holds
    the lock (it becomes the instance); otherwise the last reply (``None``, or
    a "starting or shutting down" error) or the first real answer.
    """
    log.info("The running instance does not answer; waiting for it (it may be shutting down)")
    deadline = time.monotonic() + HANDOVER_WAIT_MS / 1000.0
    reply: str | None = None
    while True:
        if lock.acquire():
            log.info("The previous instance has exited; starting")
            return True, None
        reply = ipc.send_command(request, timeout_ms=_HANDOVER_REPLY_MS)
        if reply is not None and not _is_busy(reply):
            return False, reply
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            return False, reply
        _wait_ms(min(_HANDOVER_RETRY_MS, remaining_ms))


def _wait_ms(ms: int) -> None:
    """Wait ``ms`` while still processing timers and socket events (not user input)."""
    loop = QEventLoop()
    QTimer.singleShot(max(1, int(ms)), loop.quit)
    loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)


def _warn_startup_options_ignored(args: argparse.Namespace) -> None:
    """Say which options had no effect because the request went to a running instance.

    ``--trace``, ``--camera`` and ``--backend`` are applied when an instance
    starts; a launch that only hands a request over cannot change the running
    one, and silently dropping them left users waiting for a trace file that
    never appeared.
    """
    ignored = [
        option for attribute, option in _STARTUP_ONLY_OPTIONS if getattr(args, attribute, None)
    ]
    if not ignored:
        return
    log.warning(
        "%s is already running, so %s %s not applied: %s only when %s starts. Quit it "
        "first (tray menu > Quit %s, or '%s'), then run the command again.",
        APP_NAME,
        ", ".join(ignored),
        "was" if len(ignored) == 1 else "were",
        "it applies" if len(ignored) == 1 else "they apply",
        APP_NAME,
        APP_NAME,
        cli_command_text("ctl", "quit"),
    )


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    """Apply ``--camera`` / ``--backend`` for this session.

    Returns ``{"section.field": (saved value, session value)}``.
    """
    overrides: dict[str, tuple[str, str]] = {}
    for option, key in _OVERRIDABLE.items():
        value = getattr(args, option, None)
        if not value:
            continue
        value = str(value).strip()
        saved = str(_get_setting(settings, key))
        _set_setting(settings, key, value)
        overrides[key] = (saved, value)
        # Camera values can be file paths; keep them out of the log.
        shown = value if option == "backend" else "(from the command line)"
        log.info("Setting %s overridden for this session: %s", key, shown)
    return overrides


def _get_setting(settings: Settings, key: str) -> Any:
    section, _, name = key.partition(".")
    return getattr(getattr(settings, section), name)


def _set_setting(settings: Settings, key: str, value: Any) -> None:
    section, _, name = key.partition(".")
    setattr(getattr(settings, section), name, value)


def _configure_qt_app(qt_app: QApplication) -> None:
    qt_app.setApplicationName(APP_NAME)
    qt_app.setApplicationDisplayName(APP_NAME)
    qt_app.setApplicationVersion(__version__)
    qt_app.setOrganizationName(APP_AUTHOR)
    # Wayland app id / .desktop association (packaging/linux/eye-tracker.desktop).
    qt_app.setDesktopFileName(APP_SLUG)
    # A tray app keeps running when its last window closes.
    qt_app.setQuitOnLastWindowClosed(False)
    try:
        from .ui.icons import app_icon

        qt_app.setWindowIcon(app_icon())
    except Exception:
        log.warning("Could not create the application icon", exc_info=True)


def _present(widget: QWidget) -> None:
    if widget.isMinimized():
        widget.setWindowState(widget.windowState() & ~Qt.WindowState.WindowMinimized)
    widget.show()
    widget.raise_()
    widget.activateWindow()


def _dispose_widget(widget: QWidget) -> None:
    with contextlib.suppress(Exception):
        widget.close()
    with contextlib.suppress(RuntimeError):
        widget.deleteLater()


def _noop() -> None:
    """Timer slot that only hands control to the Python interpreter."""


@contextlib.contextmanager
def _quit_on_signals(request_quit: Callable[[], None]) -> Iterator[None]:
    """Quit cleanly on Ctrl+C (and SIGTERM outside Windows) while the event loop runs.

    Python runs signal handlers only between bytecodes, which never happens while
    Qt's C++ event loop waits. A timer that calls a Python no-op hands control
    back regularly so the handler gets a chance to run.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    wanted: list[signal.Signals] = [signal.SIGINT]
    if sys.platform != "win32":
        wanted.append(signal.SIGTERM)
    previous: dict[signal.Signals, Any] = {}

    def handler(signum: int, _frame: object) -> None:
        log.info("Received signal %s; quitting", signum)
        QTimer.singleShot(0, request_quit)

    for sig in wanted:
        try:
            if signal.getsignal(sig) is signal.SIG_IGN:
                continue  # e.g. started in the background by a shell, or with nohup
            previous[sig] = signal.signal(sig, handler)
        except (OSError, ValueError):
            log.debug("Cannot handle signal %s", sig, exc_info=True)
    ticker = QTimer()
    ticker.setInterval(SIGNAL_POLL_MS)
    ticker.timeout.connect(_noop)
    ticker.start()
    try:
        yield
    finally:
        ticker.stop()
        for sig, old in previous.items():
            with contextlib.suppress(OSError, ValueError, TypeError):
                signal.signal(sig, old)


def _show_error(qt_app: QApplication, text: str) -> None:
    """Tell a user without a console that startup failed (the log has details)."""
    if qt_app.platformName() in ("offscreen", "minimal"):
        return
    QMessageBox.critical(None, APP_NAME, text)


def _isatty(stream: Any) -> bool:
    if stream is None:
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False
