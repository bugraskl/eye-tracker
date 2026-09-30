"""Logging configuration for the app and the command-line tools.

The tray app usually runs without a console (``pythonw``, a windowed build,
started at login), so the rotating log file is the only place where problems
become visible. This module therefore also routes Qt's own messages and
uncaught exceptions (main thread and worker threads) into that file.

Native libraries log through their own channels: MediaPipe uses glog/absl and
OpenCV its own logger, both writing straight to stderr. They are quietened with
environment variables, which must be set before those libraries are loaded, so
:func:`setup_logging` is meant to run first thing at startup.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import os
import sys
import threading
from types import TracebackType
from typing import Any

from . import paths

__all__ = [
    "BACKUP_COUNT",
    "LEVELS",
    "LOG_FORMAT",
    "MAX_BYTES",
    "install_qt_message_handler",
    "parse_level",
    "set_level",
    "setup_logging",
    "shutdown_logging",
]

log = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

#: Third-party Python loggers that are chatty at INFO level.
_NOISY_LOGGERS = ("absl", "PIL", "matplotlib", "asyncio")
#: Marks handlers installed by this module so repeated calls replace them.
_HANDLER_FLAG = "_eye_tracker_handler"


class _State:
    """Process-wide bookkeeping of what this module installed."""

    lock = threading.Lock()
    saved_root_level: int | None = None
    hooks_installed = False
    qt_handler: Any = None


def parse_level(level: str | int | None, default: int = logging.INFO) -> int:
    """Turn ``"debug"``, ``"INFO"``, ``20`` … into a logging level (``default`` if invalid)."""
    if isinstance(level, int) and not isinstance(level, bool):
        return level
    if isinstance(level, str):
        value = logging.getLevelName(level.strip().upper())
        if isinstance(value, int):
            return value
    return default


def setup_logging(level: str = "INFO", console: bool = False, *, log_to_file: bool = True) -> None:
    """Configure the root logger. Safe to call again (previous handlers are replaced).

    Args:
        level: ``DEBUG``, ``INFO``, ``WARNING`` or ``ERROR`` (case-insensitive;
            anything else means ``INFO``).
        console: Also log to stderr (ignored when there is no stderr, as under
            ``pythonw`` or in a windowed build).
        log_to_file: Write to the rotating log file (``paths.log_file()``,
            1 MB × 3 backups). Command-line tools that only report to the
            terminal pass ``False``.
    """
    numeric = parse_level(level)
    _quiet_native_libraries(debug=numeric <= logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT)
    new_handlers: list[logging.Handler] = []
    if log_to_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                paths.log_file(),
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
                delay=True,
            )
        except OSError as exc:
            # A read-only profile must not stop the app; fall back to stderr.
            console = True
            _write_stderr(f"Eye Tracker: cannot write the log file ({exc})")
        else:
            new_handlers.append(file_handler)
    if console and sys.stderr is not None:
        new_handlers.append(logging.StreamHandler(sys.stderr))

    root = logging.getLogger()
    with _State.lock:
        _remove_own_handlers(root)
        if _State.saved_root_level is None:
            _State.saved_root_level = root.level
        for handler in new_handlers:
            handler.setFormatter(formatter)
            setattr(handler, _HANDLER_FLAG, True)
            root.addHandler(handler)
        _install_exception_hooks()
    _apply_level(numeric)
    logging.captureWarnings(True)


def set_level(level: str | int) -> None:
    """Change the verbosity after :func:`setup_logging` (e.g. once settings are loaded)."""
    numeric = parse_level(level)
    _apply_level(numeric)
    _quiet_native_libraries(debug=numeric <= logging.DEBUG)


def shutdown_logging() -> None:
    """Flush and remove the handlers installed by :func:`setup_logging`.

    Restores the root level found before the first setup. Used at exit and by
    tests (an open log file would keep a temporary directory locked on Windows).
    """
    root = logging.getLogger()
    with _State.lock:
        _remove_own_handlers(root)
        if _State.saved_root_level is not None:
            root.setLevel(_State.saved_root_level)
            _State.saved_root_level = None
    logging.captureWarnings(False)


def install_qt_message_handler() -> None:
    """Send Qt's qDebug/qWarning/… output to the ``qt`` logger instead of stderr.

    Needs PySide6; does nothing if it cannot be imported. Idempotent.
    """
    if _State.qt_handler is not None:
        return
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except ImportError:
        return
    qt_log = logging.getLogger("qt")
    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(msg_type: Any, context: Any, message: str) -> None:
        category = getattr(context, "category", None)
        prefix = f"[{category}] " if category and category != "default" else ""
        qt_log.log(levels.get(msg_type, logging.WARNING), "%s%s", prefix, message)

    # Keep a reference: Qt only stores a pointer to the Python callable.
    _State.qt_handler = handler
    qInstallMessageHandler(handler)


# ---------------------------------------------------------------------- internals
def _apply_level(numeric: int) -> None:
    logging.getLogger().setLevel(numeric)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(numeric, logging.WARNING))


def _quiet_native_libraries(*, debug: bool) -> None:
    """Silence MediaPipe/TensorFlow Lite and OpenCV stderr chatter unless debugging."""
    if debug:
        return
    # glog/absl (MediaPipe) and TFLite read these when they are first loaded.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # OpenCV reads this on first use. Without it, probing a camera index that does
    # not exist prints "[ WARN ] VIDEOIO(DSHOW) ... can't be used to capture by index".
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    # Importing cv2 costs ~0.4 s, so it is only adjusted here when already loaded
    # (the environment variable covers a later import).
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:  # an OpenCV build without the logging module
            log.debug("Could not lower the OpenCV log level", exc_info=True)


def _remove_own_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            root.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()


def _install_exception_hooks() -> None:
    """Log uncaught exceptions; without a console they would otherwise vanish."""
    if _State.hooks_installed:
        return
    _State.hooks_installed = True
    crash_log = logging.getLogger("eye_tracker.crash")
    previous_hook = sys.excepthook
    previous_thread_hook = threading.excepthook

    def excepthook(
        exc_type: type[BaseException],
        exc: BaseException,
        tb: TracebackType | None,
    ) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            crash_log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
        if sys.stderr is not None:
            previous_hook(exc_type, exc, tb)

    def thread_excepthook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not SystemExit and args.exc_value is not None:
            name = args.thread.name if args.thread is not None else "?"
            crash_log.critical(
                "Unhandled exception in thread %s",
                name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        if sys.stderr is not None:
            previous_thread_hook(args)

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook


def _write_stderr(message: str) -> None:
    if sys.stderr is not None:
        with contextlib.suppress(Exception):
            sys.stderr.write(message + "\n")
