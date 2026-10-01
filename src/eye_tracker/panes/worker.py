"""Talk to pane providers off the GUI thread.

Providers run command-line tools that can take tens of milliseconds, or hang
until their timeout, so :class:`PaneWorker` runs them on one background thread
(like the controller's screen-lock thread) and hands the results back through
the Qt signals :attr:`PaneWorker.detected` and :attr:`PaneWorker.focused`.
Emitted from the worker thread, they reach slots of objects on the GUI thread
as queued calls.

Requests do not pile up: a new detection request replaces one that has not
started yet, and a focus request goes before any detection. A provider whose
calls fail (raise, or take longer than the call timeout) :data:`FAILURE_LIMIT`
times in a row is switched off for the rest of the session.

Providers may hold system resources (UI Automation COM objects, which belong
to the thread that made them). Their ``close()``, where they have one, runs on
the worker's thread: for the old providers when :meth:`PaneWorker.set_registry`
replaces them, and for the current ones when the thread ends.

``synchronous=True`` runs every request on the calling thread instead, for tests.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from PySide6.QtCore import QObject, Signal

from ..types import AppIdentity, WindowRef
from .registry import PaneRegistry
from .types import Pane, PaneProvider, PaneSnapshot

log = logging.getLogger(__name__)

#: Consecutive failures after which a provider is off for the session.
FAILURE_LIMIT = 3
#: A provider call taking longer than this counts as failed (its result is dropped).
CALL_TIMEOUT_S = 2.5
#: How long :meth:`PaneWorker.stop` waits for a call in progress.
JOIN_TIMEOUT_S = 3.0


@dataclass(frozen=True, slots=True)
class DetectResult:
    """Answer to :meth:`PaneWorker.request_detect`."""

    window_handle: Any
    #: The panes found (``taken_at`` stamped with the worker's clock), or ``None``.
    snapshot: PaneSnapshot | None
    request_id: int


@dataclass(frozen=True, slots=True)
class FocusResult:
    """Answer to :meth:`PaneWorker.request_focus`."""

    window_handle: Any
    pane_id: Any
    ok: bool


@dataclass(slots=True)
class _Detect:
    ref: WindowRef
    app: AppIdentity
    request_id: int


@dataclass(slots=True)
class _Focus:
    ref: WindowRef
    pane: Pane


class PaneWorker(QObject):
    """Single background thread for pane detection and focus requests."""

    #: A detection finished (:class:`DetectResult`).
    detected = Signal(object)
    #: A focus request finished (:class:`FocusResult`).
    focused = Signal(object)

    def __init__(
        self,
        registry: PaneRegistry,
        *,
        clock: Callable[[], float] = time.monotonic,
        synchronous: bool = False,
        call_timeout_s: float = CALL_TIMEOUT_S,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._clock = clock
        self._synchronous = synchronous
        self._call_timeout = call_timeout_s
        self._cond = threading.Condition()
        self._detect: _Detect | None = None
        self._focus: _Focus | None = None
        self._stopping = False
        #: Registries replaced by set_registry whose providers are still to be closed.
        self._retired: list[PaneRegistry] = []
        self._thread: threading.Thread | None = None
        self._failures: dict[str, int] = {}
        self._disabled: set[str] = set()
        self._next_id = 0

    # ------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Start the thread (a no-op when synchronous or already running)."""
        if self._synchronous or self.running:
            return
        with self._cond:
            self._stopping = False
            self._detect = None  # asked while stopped: stale by now
            self._focus = None
        thread = threading.Thread(target=self._run, name="eye-tracker-panes", daemon=True)
        self._thread = thread
        thread.start()

    def request_stop(self) -> None:
        """Drop pending requests and tell the thread to end, without waiting.

        Nothing is delivered after this. The thread finishes a provider call in
        progress, closes the providers and ends; without a running thread (or
        when synchronous) the providers are closed here.
        """
        with self._cond:
            self._stopping = True
            self._detect = None
            self._focus = None
            self._cond.notify_all()
        thread = self._thread
        if thread is None or not thread.is_alive():
            self._close_retired()
            self._registry.close()

    def stop(self, timeout: float = JOIN_TIMEOUT_S) -> None:
        """:meth:`request_stop`, then wait up to ``timeout`` for the thread to end."""
        self.request_stop()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                log.warning("A pane provider call did not finish while stopping")

    def set_registry(self, registry: PaneRegistry) -> None:
        """Use other providers from the next request on (settings changed).

        The replaced providers are closed on the worker's thread before its next
        request (at once when synchronous).
        """
        with self._cond:
            old, self._registry = self._registry, registry
            self._detect = None
            if old is registry:
                return
            self._retired.append(old)
            self._cond.notify_all()
        if self._synchronous:
            self._close_retired()

    @property
    def disabled_providers(self) -> frozenset[str]:
        with self._cond:
            return frozenset(self._disabled)

    # ------------------------------------------------------------- requests
    def request_detect(self, ref: WindowRef, app: AppIdentity) -> int:
        """Find the panes of ``ref``; the answer comes through :attr:`detected`.

        Replaces a detection request that has not started yet. Returns the
        request's id (echoed in :class:`DetectResult`).
        """
        with self._cond:
            self._next_id += 1
            job = _Detect(ref, app, self._next_id)
            if not self._synchronous:
                self._detect = job
                self._cond.notify_all()
                return job.request_id
        self._do_detect(job)
        return job.request_id

    def request_focus(self, ref: WindowRef, pane: Pane) -> None:
        """Focus ``pane``; the answer comes through :attr:`focused`."""
        job = _Focus(ref, pane)
        if self._synchronous:
            self._do_focus(job)
            return
        with self._cond:
            self._focus = job
            self._cond.notify_all()

    # --------------------------------------------------------------- thread
    def _run(self) -> None:
        try:
            self._serve()
        finally:
            # On this thread: UI Automation objects belong to the thread that made them.
            self._close_retired()
            with self._cond:
                registry = self._registry
            registry.close()

    def _serve(self) -> None:
        while True:
            self._close_retired()
            with self._cond:
                while (
                    not self._stopping
                    and self._detect is None
                    and self._focus is None
                    and not self._retired
                ):
                    self._cond.wait()
                if self._stopping:
                    return
                if self._retired:
                    continue  # close them first
                job: _Detect | _Focus
                if self._focus is not None:
                    job, self._focus = self._focus, None
                else:
                    assert self._detect is not None
                    job, self._detect = self._detect, None
            try:
                if isinstance(job, _Focus):
                    self._do_focus(job)
                else:
                    self._do_detect(job)
            except Exception:  # never let the thread die
                log.warning("Pane request failed", exc_info=True)

    def _do_detect(self, job: _Detect) -> None:
        with self._cond:
            registry = self._registry
        snapshot: PaneSnapshot | None = None
        for provider in registry.providers_for(job.app):
            if self._is_disabled(provider.name):
                continue
            found = self._call(provider, functools.partial(provider.detect, job.ref, job.app))
            if isinstance(found, PaneSnapshot) and len(found.panes) >= 2:
                snapshot = replace(found, window_handle=job.ref.handle, taken_at=self._clock())
                break
        self._emit(self.detected, DetectResult(job.ref.handle, snapshot, job.request_id))

    def _do_focus(self, job: _Focus) -> None:
        with self._cond:
            registry = self._registry
        provider = registry.provider(job.pane.provider)
        ok = False
        if provider is not None and not self._is_disabled(provider.name):
            ok = self._call(provider, functools.partial(provider.focus, job.ref, job.pane)) is True
        self._emit(self.focused, FocusResult(job.ref.handle, job.pane.id, ok))

    def _call(self, provider: PaneProvider, fn: Callable[[], Any]) -> Any:
        """Run one provider call, counting failures; ``None`` when it failed."""
        started = time.monotonic()
        try:
            result = fn()
        except Exception as exc:
            self._failed(provider.name, str(exc) or type(exc).__name__)
            return None
        elapsed = time.monotonic() - started
        if elapsed > self._call_timeout:
            self._failed(provider.name, f"took {elapsed:.1f} s")
            return None
        with self._cond:
            self._failures.pop(provider.name, None)
        return result

    def _failed(self, name: str, why: str) -> None:
        with self._cond:
            count = self._failures.get(name, 0) + 1
            self._failures[name] = count
            if count >= FAILURE_LIMIT:
                self._disabled.add(name)
        log.debug("Pane provider %s failed (%d in a row): %s", name, count, why)
        if count == FAILURE_LIMIT:
            log.info("Pane provider %s keeps failing; off until the next start", name)

    def _close_retired(self) -> None:
        with self._cond:
            retired, self._retired = self._retired, []
        for registry in retired:
            registry.close()

    def _is_disabled(self, name: str) -> bool:
        with self._cond:
            return name in self._disabled

    def _emit(self, signal: Any, value: object) -> None:
        with self._cond:
            if self._stopping and not self._synchronous:
                return
        with contextlib.suppress(RuntimeError):  # deleted after a stop timed out
            signal.emit(value)
