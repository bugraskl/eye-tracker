"""Windows Terminal panes, through UI Automation (Windows only).

Windows Terminal has no command-line interface that lists or focuses panes,
but every pane is a ``TermControl`` element in its UI Automation tree. Looking
up the ``TermControl`` descendants of the window gives each pane's bounding
rectangle (physical pixels in global coordinates, like the rest of the app on
Windows, which is per-monitor DPI aware) and whether it has the keyboard focus;
``SetFocus`` on another one makes it the active pane. Nothing is typed or
clicked, and neither names nor text of any element are read.

Only the tab on screen has ``TermControl`` elements in the tree; off-screen
ones (and empty rectangles) are ignored all the same.

Pane ids are UI Automation runtime ids (``GetRuntimeId``), which stay the same
for as long as the pane exists, so the pane decider can follow a pane from one
snapshot to the next. If a runtime id cannot be read, the pane's rectangle
stands in for it (``("rect", x, y, w, h)``): stable while the layout is.

Both calls run only while the window is in the foreground: Windows Terminal
reports the keyboard focus only then, and ``SetFocus`` on a background window
could bring it to the front. The controller only asks about the foreground
window anyway, but the pane worker answers a little later.

One provider serves one window: when Windows Terminal shows two or more panes
they are used and tmux running inside one of them is not looked at; with a
single Windows Terminal pane the tmux provider gets its turn (see
``..registry``). Following tmux panes nested inside Windows Terminal panes is
left for later.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from typing import Any, Protocol

from ...platform.win_uia import UiaElement, UiAutomation
from ...types import AppIdentity, Rect, WindowRef
from ..types import Pane, PaneError, PaneSnapshot

log = logging.getLogger(__name__)

#: Process names (see ``platform.base.process_basename``) of Windows Terminal.
TERMINAL_PROCESSES: frozenset[str] = frozenset({"windowsterminal"})
#: Class of Windows Terminal's top-level windows.
WINDOW_CLASS = "CASCADIA_HOSTING_WINDOW_CLASS"
#: UI Automation class name of one terminal pane.
PANE_CLASS = "TermControl"


class UiaClient(Protocol):
    """The UI Automation calls the provider makes (``platform.win_uia.UiAutomation``)."""

    def find(self, hwnd: int, class_name: str) -> list[UiaElement]: ...

    def focus(self, hwnd: int, class_name: str, match: Callable[[UiaElement], bool]) -> bool: ...

    def foreground_window(self) -> int | None: ...


def pane_id(element: UiaElement) -> Hashable:
    """The pane id of a ``TermControl``: its runtime id, else its rectangle."""
    if element.runtime_id:
        return element.runtime_id
    rect = element.rect
    if rect is None:
        return None
    return ("rect", rect.x, rect.y, rect.w, rect.h)


def _clip(rect: Rect, bounds: Rect | None) -> Rect | None:
    """``rect`` cut to ``bounds``; ``None`` if nothing is left."""
    if bounds is None:
        return rect
    x0, y0 = max(rect.x, bounds.x), max(rect.y, bounds.y)
    x1, y1 = min(rect.right, bounds.right), min(rect.bottom, bounds.bottom)
    if x1 <= x0 or y1 <= y0:
        return None
    return Rect(x0, y0, x1 - x0, y1 - y0)


def _hwnd(ref: WindowRef) -> int | None:
    handle = ref.handle
    if isinstance(handle, int) and not isinstance(handle, bool) and handle > 0:
        return handle
    return None


class WindowsTerminalProvider:
    """Panes of the tab a Windows Terminal window shows."""

    name = "windows_terminal"

    def __init__(
        self,
        *,
        client_rect: Callable[[WindowRef], Rect | None],
        runner: Any = None,  # registry interface; this provider runs no commands
        client: UiaClient | None = None,
    ) -> None:
        self._client_rect = client_rect
        self._client = client

    def applies(self, app: AppIdentity) -> bool:
        return app.process.strip().lower() in TERMINAL_PROCESSES and app.app_id == WINDOW_CLASS

    def detect(self, ref: WindowRef, app: AppIdentity) -> PaneSnapshot | None:
        hwnd = _hwnd(ref)
        if hwnd is None:
            return None
        client = self._uia()
        try:
            if client.foreground_window() != hwnd:
                return None
            elements = client.find(hwnd, PANE_CLASS)
        except OSError as exc:
            raise PaneError(f"UI Automation: {exc}") from exc
        bounds = self._client_rect(ref) or ref.rect
        panes: list[Pane] = []
        seen: set[Hashable] = set()
        for element in elements:
            if element.offscreen or element.rect is None:
                continue
            rect = _clip(element.rect, bounds)
            ident = pane_id(element)
            if rect is None or ident is None or ident in seen:
                continue
            seen.add(ident)
            panes.append(Pane(ident, rect, element.has_keyboard_focus, self.name))
        if len(panes) < 2:
            return None
        panes.sort(key=lambda p: (p.rect.y, p.rect.x))
        return PaneSnapshot(ref.handle, tuple(panes), 0.0)

    def focus(self, ref: WindowRef, pane: Pane) -> bool:
        hwnd = _hwnd(ref)
        if hwnd is None or pane.provider != self.name or pane.id is None:
            return False
        client = self._uia()

        def match(element: UiaElement) -> bool:
            return not element.offscreen and pane_id(element) == pane.id

        try:
            if client.foreground_window() != hwnd:
                return False  # SetFocus could bring a background window to the front
            return bool(client.focus(hwnd, PANE_CLASS, match))
        except OSError as exc:
            raise PaneError(f"UI Automation: {exc}") from exc

    def _uia(self) -> UiaClient:
        """The UI Automation client, created on first use (on the worker thread)."""
        if self._client is None:
            self._client = UiAutomation()
        return self._client
