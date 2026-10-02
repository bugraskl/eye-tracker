"""The update service of the running app: when to look, what to tell, what to install.

It lives next to the controller, not inside it: the engine that watches the
camera never touches the network. The service

* looks for a newer release **only** when the user turned the check on (once a
  day, the first time a minute and a half after the app started) or asked for it
  (tray menu, About);
* tells about a version once (a tray notification), then only keeps the menu
  entry "Update to X.Y.Z…";
* installs nothing by itself: :meth:`UpdateService.install` is called after the
  user agreed in the update window.

Work runs on a background thread; the state arrives on the main thread through
:attr:`UpdateService.state_changed`. What it remembers between runs (the time of
the last check and the version it already told about) is kept in ``updates.json``
in the data folder.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from .. import __version__, paths
from ..config import atomic_write_text
from . import installer
from .fetch import Cancelled, Fetcher, NetworkError, supported
from .release import (
    LATEST_RELEASE_URL,
    MAX_API_BYTES,
    ReleaseInfo,
    is_newer,
    parse_release,
)

log = logging.getLogger(__name__)

#: First automatic check, after the app started (the start itself stays quiet).
FIRST_CHECK_DELAY_S = 90.0
#: The automatic check is due again after this long.
CHECK_INTERVAL_S = 24 * 3600.0
#: How often the service looks whether a check is due (a laptop may sleep for days).
TICK_INTERVAL_S = 3 * 3600.0
#: Progress is passed on at most this often.
PROGRESS_INTERVAL_S = 0.2


class Phase(Enum):
    IDLE = "idle"
    CHECKING = "checking"
    UP_TO_DATE = "up_to_date"
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class UpdateState:
    phase: Phase = Phase.IDLE
    #: The newest release found (set from ``AVAILABLE`` on, also while installing).
    release: ReleaseInfo | None = None
    message: str = ""
    done: int = 0
    total: int = 0
    #: The user asked for the check that produced this state (a result is always shown).
    manual: bool = False
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CheckResult:
    release: ReleaseInfo | None
    newer: bool


def check_for_update(fetcher: Fetcher, running: str = __version__) -> CheckResult:
    """Ask GitHub for the latest release and compare it with ``running``."""
    body = fetcher.get(
        LATEST_RELEASE_URL, max_bytes=MAX_API_BYTES, accept="application/vnd.github+json"
    )
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise NetworkError("GitHub's answer could not be read.") from exc
    release = parse_release(payload)
    if release is None:
        raise NetworkError("GitHub's answer is not a release this app understands.")
    return CheckResult(release, is_newer(release, running))


class _Memory:
    """The small file that keeps the last check and the version told about."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.last_check = 0.0
        self.notified = ""
        with contextlib.suppress(OSError, ValueError):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                last = data.get("last_check")
                if isinstance(last, (int, float)) and not isinstance(last, bool):
                    self.last_check = float(last)
                if isinstance(data.get("notified"), str):
                    self.notified = data["notified"]

    def save(self) -> None:
        text = json.dumps({"last_check": self.last_check, "notified": self.notified}) + "\n"
        try:
            atomic_write_text(self._path, text)
        except OSError:
            log.debug("Could not save the update memory", exc_info=True)


class UpdateService(QObject):
    """See the module docstring. Create it on the main thread."""

    #: Every change of the state (an :class:`UpdateState`), on the main thread.
    state_changed = Signal(object)
    #: A version the user was not told about yet was found by an automatic check
    #: (the :class:`ReleaseInfo`): show a notification once.
    announce = Signal(object)

    _finished = Signal(object)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        fetcher_factory: Callable[[], Fetcher] = Fetcher,
        clock: Callable[[], float] = time.time,
        running_version: str = __version__,
        memory_path: Path | None = None,
        kind: str | None = None,
        launcher: Callable[[Path], None] = installer.launch_installer,
    ) -> None:
        super().__init__(parent)
        self._fetcher_factory = fetcher_factory
        self._clock = clock
        self._running = running_version
        self._launcher = launcher
        self._kind = kind if kind is not None else installer.install_kind()
        self._memory = _Memory(memory_path or paths.data_dir() / "updates.json")
        self._auto = False
        self._state = UpdateState()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._last_progress = 0.0

        self._finished.connect(self._on_finished, Qt.ConnectionType.QueuedConnection)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)

    # ---------------------------------------------------------------- queries
    @property
    def state(self) -> UpdateState:
        return self._state

    @property
    def available(self) -> bool:
        """Whether this platform can look for updates at all."""
        return supported()

    @property
    def kind(self) -> str:
        """``installer.KIND_SETUP`` (the app can install it), ``KIND_PORTABLE`` or
        ``KIND_UNSUPPORTED``."""
        return self._kind

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def can_install(self) -> bool:
        release = self._state.release
        return (
            self._kind == installer.KIND_SETUP
            and release is not None
            and release.installable
            and self._state.phase in (Phase.AVAILABLE, Phase.FAILED)
        )

    # --------------------------------------------------------------- lifecycle
    def start(self, auto: bool) -> None:
        """Begin: clean up old downloads and, when ``auto``, schedule the first check."""
        installer.remove_downloads()
        self.set_auto(auto)

    def set_auto(self, enabled: bool) -> None:
        """Turn the daily check on or off (the setting ``updates.check``)."""
        self._auto = bool(enabled) and self.available
        if not self._auto:
            self._timer.stop()
            return
        if not self._timer.isActive():
            self._timer.start(int(FIRST_CHECK_DELAY_S * 1000))

    def shutdown(self) -> None:
        self._timer.stop()
        self._cancel.set()

    # ----------------------------------------------------------------- actions
    def check_now(self) -> bool:
        """Look for an update because the user asked. Returns ``False`` when busy
        or unavailable on this platform."""
        return self._begin_check(manual=True)

    def install(self) -> bool:
        """Download, verify and start the installer of the release found. Returns
        ``False`` when nothing can be installed right now."""
        release = self._state.release
        if self.busy or not self.can_install or release is None:
            return False
        self._cancel.clear()
        self._set(UpdateState(Phase.DOWNLOADING, release, "Downloading…", manual=True))
        self._start_thread(self._download, release)
        return True

    def cancel(self) -> None:
        """Stop a running download."""
        self._cancel.set()

    # --------------------------------------------------------------- internals
    def _tick(self) -> None:
        if not self._auto:
            return
        now = self._clock()
        due = now - self._memory.last_check >= CHECK_INTERVAL_S or self._memory.last_check > now
        if due and not self.busy:
            self._begin_check(manual=False)
        self._timer.start(int(TICK_INTERVAL_S * 1000))

    def _begin_check(self, *, manual: bool) -> bool:
        if self.busy or not self.available:
            return False
        if self._state.phase in (Phase.CHECKING, Phase.DOWNLOADING, Phase.INSTALLING):
            return False  # the result of the one before is still on its way
        self._memory.last_check = self._clock()
        self._memory.save()
        previous = self._state.release
        self._set(UpdateState(Phase.CHECKING, previous, "Checking…", manual=manual))
        self._start_thread(self._check, manual)
        return True

    def _start_thread(self, target: Callable[..., None], *args: object) -> None:
        self._thread = threading.Thread(
            target=target, args=args, name="eye-tracker-update", daemon=True
        )
        self._thread.start()

    def _set(self, state: UpdateState) -> None:
        self._state = state
        self.state_changed.emit(state)

    def _check(self, manual: bool) -> None:
        try:
            with self._fetcher_factory() as fetcher:
                result = check_for_update(fetcher, self._running)
        except (OSError, ValueError) as exc:
            log.info("Update check failed: %s", exc)
            self._finished.emit(UpdateState(Phase.FAILED, None, _reason(exc), manual=manual))
            return
        except Exception:
            log.exception("Update check failed")
            self._finished.emit(
                UpdateState(Phase.FAILED, None, "The check failed unexpectedly.", manual=manual)
            )
            return
        phase = Phase.AVAILABLE if result.newer else Phase.UP_TO_DATE
        self._finished.emit(UpdateState(phase, result.release, manual=manual))

    def _download(self, release: ReleaseInfo) -> None:
        def progress(done: int, total: int) -> None:
            now = time.monotonic()
            if now - self._last_progress >= PROGRESS_INTERVAL_S:
                self._last_progress = now
                self._finished.emit(
                    UpdateState(Phase.DOWNLOADING, release, "Downloading…", done, total, True)
                )

        try:
            with self._fetcher_factory() as fetcher:
                path = installer.download_installer(
                    release, fetcher, progress=progress, cancelled=self._cancel.is_set
                )
        except Cancelled:
            self._finished.emit(UpdateState(Phase.AVAILABLE, release, manual=True))
            return
        except (OSError, ValueError, installer.UpdateError) as exc:
            log.warning("Update download failed: %s", exc)
            self._finished.emit(UpdateState(Phase.FAILED, release, _reason(exc), manual=True))
            return
        except Exception:
            log.exception("Update download failed")
            self._finished.emit(
                UpdateState(Phase.FAILED, release, "The download failed unexpectedly.", manual=True)
            )
            return
        self._finished.emit(
            UpdateState(
                Phase.INSTALLING, release, "Installing…", manual=True, extra={"path": str(path)}
            )
        )

    def _on_finished(self, state: UpdateState) -> None:
        """A result from the worker thread, on the main thread."""
        if state.phase is Phase.DOWNLOADING:
            if self._state.phase is Phase.DOWNLOADING:
                self._set(state)  # progress; a cancelled download is no longer shown
            return
        if state.phase is Phase.INSTALLING:
            self._set(state)
            self._start_installer(state)
            return
        self._set(state)
        release = state.release
        told = release is not None and self._memory.notified == release.version
        if state.phase is Phase.AVAILABLE and release is not None and not told:
            self._memory.notified = release.version
            self._memory.save()
            if not state.manual:  # a manual check shows its result in the window
                self.announce.emit(release)

    def _start_installer(self, state: UpdateState) -> None:
        path = Path(state.extra.get("path", ""))
        try:
            self._launcher(path)
        except (OSError, installer.UpdateError) as exc:
            log.warning("Could not start the installer: %s", exc)
            self._set(UpdateState(Phase.FAILED, state.release, _reason(exc), manual=True))


def _reason(exc: BaseException) -> str:
    text = str(exc).strip()
    return text if text else "Something went wrong."
