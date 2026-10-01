"""Value types shared by the split-pane providers, worker, decider and controller."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Protocol

from ..types import AppIdentity, Rect, WindowRef


class PaneError(Exception):
    """A provider could not talk to its terminal (timeout, unreadable output, ...).

    Counts as a failure of the provider; "this window has no panes of mine" is
    not an error but ``None`` from :meth:`PaneProvider.detect`.
    """


@dataclass(frozen=True, slots=True)
class Pane:
    """One split pane of a terminal window.

    ``id`` is the provider's own identifier (tmux ``"%3"``, a WezTerm pane id);
    ``rect`` is where the pane is on the screen, in the global coordinates of
    :mod:`eye_tracker.types`. Titles, commands and working directories are never
    part of it.
    """

    id: Hashable
    rect: Rect
    focused: bool
    provider: str


@dataclass(frozen=True, slots=True)
class PaneSnapshot:
    """The panes of one window as a provider saw them at ``taken_at`` (monotonic s)."""

    window_handle: Any
    panes: tuple[Pane, ...]
    taken_at: float

    @property
    def focused(self) -> Pane | None:
        """The pane with keyboard focus (``None`` if the provider could not tell)."""
        return next((p for p in self.panes if p.focused), None)

    @property
    def provider(self) -> str | None:
        return self.panes[0].provider if self.panes else None

    def pane(self, pane_id: Hashable) -> Pane | None:
        return next((p for p in self.panes if p.id == pane_id), None)

    def with_focus(self, pane_id: Hashable) -> PaneSnapshot:
        """A copy in which ``pane_id`` has the focus (after we gave it focus)."""
        panes = tuple(Pane(p.id, p.rect, p.id == pane_id, p.provider) for p in self.panes)
        return PaneSnapshot(self.window_handle, panes, self.taken_at)


class PaneProvider(Protocol):
    """Finds and focuses the split panes of one kind of terminal.

    Providers run on the pane worker's thread (:mod:`.worker`), never on the GUI
    thread, and may block for at most their command timeout. They must never
    synthesise input: they talk to the terminal or multiplexer through its own
    command-line interface (or, later, accessibility APIs).
    """

    #: Short name (``"tmux"``) used in logs, ``status`` and the trace.
    name: str

    def applies(self, app: AppIdentity) -> bool:
        """Whether windows of ``app`` may have panes of this kind. Must be cheap."""
        ...

    def detect(self, ref: WindowRef, app: AppIdentity) -> PaneSnapshot | None:
        """The panes of the window, ``None`` if it has fewer than two of this kind
        (then the next provider may look: a single WezTerm pane can run tmux).

        Raises :class:`PaneError` when the terminal could not be asked.
        """
        ...

    def focus(self, ref: WindowRef, pane: Pane) -> bool:
        """Give ``pane`` the keyboard focus inside its window; False if that failed."""
        ...
