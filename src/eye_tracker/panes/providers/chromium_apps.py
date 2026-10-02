"""Side-by-side sessions of desktop apps built on Chromium, through UI Automation.

Opt-in (``panes.desktop_apps``) and Windows only. Electron and Chromium
applications are on the registry's hard deny-list (``..registry``) because
asking them about their contents makes them build an accessibility tree, which
costs them CPU and memory. With the setting on, exactly the apps that have an
:class:`AppProfile` here (process name *and* window class) skip the deny-list
and are asked by this provider, and by no other. Every other Chromium window
(VS Code, Cursor, browsers, apps without a profile) stays denied.

Profiles
--------
Class names are the HTML class attribute, split on whitespace and matched token
by token (never as substrings). A profile finds its panes in one of two ways:

* by a **pane token**: every ``Group`` whose class has the token is a pane. The
  Claude desktop app shows two chat sessions side by side, each a group with the
  token ``dframe-pane``.
* by a **container prefix**: the direct ``Group`` children of the first group
  with a class token starting with the prefix are the panes (CSS-module classes
  carry a build hash, so only the prefix is stable). The ChatGPT desktop app
  (which also hosts Codex) shows the main conversation and a side chat or side
  panel as the children of ``_MainContentSurface_<hash>``.

Either way a pane must be at least :attr:`AppProfile.min_pane` pixels in both
directions, which leaves out small overlays. Each session has a message box: an
``Edit`` whose class has the token ``ProseMirror``. Names are never used (they
are localised, and they are what the user wrote).

The walk (``detect``)
---------------------
The control view of the window is walked step by step (first child, next
sibling), never as a whole: a full walk reads ~630 elements of the Claude
window in ~0.7 s, ~420 of the ChatGPT window in ~0.33 s.

1. From the window element, depth first through the native views (any control
   type except ``Document``, at most :data:`NATIVE_DEPTH` levels) to the
   ``Document`` whose automation id is ``RootWebArea``: the web content. Other
   documents are not entered. A window may hold more than one such document
   (the Claude app does, the first one nearly empty): they are tried in tree
   order until one has panes.
2. From there through ``Group`` elements only, at most :data:`WEB_DEPTH` levels
   below the document. A group whose class has the profile's pane token is a
   pane: it is not entered. A group with a token starting with the profile's
   container prefix is the container: its children (one step each) are the
   panes and the walk ends there. A group with one of the profile's skip tokens
   (the sidebar) is not entered. Once a group's children included panes, the
   walk ends after that group.

Every step of the walk (a first-child or next-sibling call) counts against
:data:`NODE_BUDGET`; when it is used up the walk stops with what it found. On
trees shaped like the measured ones, a detection takes about 25 steps and 35
property reads for Claude, 15 and 30 for ChatGPT (a focus a few more). Per
element only the control type is read, plus the class name of groups and
editors and the automation id of documents; of the panes their rectangle,
visibility and runtime id. Names, values and text are never read.

Panes are the pane groups that are not off-screen and whose rectangle, cut to
the window, is at least :attr:`AppProfile.min_pane` large, at least two of
them. The focused pane is the one containing the element with the keyboard
focus (``GetFocusedElement``): its centre must lie in the pane and it must not
be larger than the pane, so the whole page having the focus means no pane has
it. The focused element counts
only if it belongs to the app's process. Pane ids are runtime ids, else the
pane's rectangle (as for Windows Terminal).

Chromium builds its accessibility tree only once a client asks: the first walk
often finds only the native views, or the web content without panes. That is
"no panes" (``None``), not a failure; the next refresh finds them.

Focus
-----
The pane is found again by the same walk, then its message box by a walk of
the pane's subtree from the last child backwards (the message box is at the
bottom, after a conversation of any length), at most :data:`COMPOSER_DEPTH`
levels and :data:`COMPOSER_BUDGET` elements, not entering ``Edit`` elements.
The first ``Edit`` with the profile's composer token is the message box; if
there is none, the first keyboard-focusable ``Edit`` seen. ``SetFocus`` on it
moves the keyboard focus into that session. Nothing is typed or clicked.

Both calls run only while the window is in the foreground (like Windows
Terminal): ``SetFocus`` on a background window could bring it to the front.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from ...platform.win_uia import (
    UIA_DOCUMENT_CONTROL_TYPE_ID,
    UIA_EDIT_CONTROL_TYPE_ID,
    UIA_GROUP_CONTROL_TYPE_ID,
    UiAutomation,
)
from ...types import AppIdentity, Rect, WindowRef
from ..types import Pane, PaneError, PaneSnapshot
from .windows_terminal import _clip, _hwnd

log = logging.getLogger(__name__)

#: Name of the provider (``Pane.provider``, logs, ``status``).
PROVIDER_NAME = "desktop_apps"
#: Automation id of the document holding a Chromium window's web content.
ROOT_WEB_AREA = "RootWebArea"
#: Most elements one detection steps to (the window walk of ``focus`` too).
NODE_BUDGET = 400
#: Most elements the search for a message box steps to, inside one pane.
COMPOSER_BUDGET = 400
#: Deepest level below the window searched for the web content document.
NATIVE_DEPTH = 14
#: Deepest level below the web content document searched for panes.
WEB_DEPTH = 8
#: Deepest level below a pane searched for its message box.
COMPOSER_DEPTH = 24


@dataclass(frozen=True, slots=True)
class AppProfile:
    """How to find the side-by-side sessions of one desktop app.

    Exactly one of ``pane_token`` and ``container_prefix`` is set.
    """

    #: Short name used in logs.
    name: str
    #: Process names (``platform.base.process_basename``) of the app.
    processes: frozenset[str]
    #: Class of the app's top-level windows.
    window_class: str
    #: Class token of the session's message box (an ``Edit``).
    composer_token: str
    #: Class token of a ``Group`` that is one session pane.
    pane_token: str | None = None
    #: Prefix of a class token of the ``Group`` whose ``Group`` children are the panes.
    container_prefix: str | None = None
    #: Class tokens of groups never entered while looking for panes.
    skip_tokens: frozenset[str] = field(default_factory=frozenset)
    #: Smallest pane (width, height) in pixels; smaller children are overlays.
    min_pane: tuple[int, int] = (200, 200)

    def __post_init__(self) -> None:
        if (self.pane_token is None) == (self.container_prefix is None):
            raise ValueError("an AppProfile needs exactly one of pane_token, container_prefix")

    def matches(self, app: AppIdentity) -> bool:
        return app.process.strip().lower() in self.processes and app.app_id == self.window_class

    def is_container(self, tokens: frozenset[str]) -> bool:
        """Whether a group with these class tokens holds the panes."""
        prefix = self.container_prefix
        if not prefix:
            return False
        return any(t.startswith(prefix) for t in tokens)

    def large_enough(self, rect: Rect) -> bool:
        return rect.w >= self.min_pane[0] and rect.h >= self.min_pane[1]


#: The Claude desktop app (Windows).
CLAUDE = AppProfile(
    name="claude",
    processes=frozenset({"claude"}),
    window_class="Chrome_WidgetWin_1",
    pane_token="dframe-pane",
    composer_token="ProseMirror",
    skip_tokens=frozenset({"dframe-sidebar"}),
)

#: The ChatGPT desktop app (Windows), which also hosts Codex.
CHATGPT = AppProfile(
    name="chatgpt",
    processes=frozenset({"chatgpt"}),
    window_class="Chrome_WidgetWin_1",
    container_prefix="_MainContentSurface_",
    composer_token="ProseMirror",
    skip_tokens=frozenset({"app-shell-left-panel"}),
)

#: Every app with a profile; only these ever skip the registry's deny-list.
PROFILES: tuple[AppProfile, ...] = (CLAUDE, CHATGPT)


def profile_for(app: AppIdentity | None) -> AppProfile | None:
    """The profile of ``app`` (process name and window class), if it has one."""
    if app is None:
        return None
    return next((p for p in PROFILES if p.matches(app)), None)


def class_tokens(class_name: str) -> frozenset[str]:
    """The whitespace-separated tokens of an element's class name."""
    return frozenset(class_name.split())


class ElementView(Protocol):
    """A walk of one window's control view (``platform.win_uia.ControlView``).

    Nodes are opaque; they stay valid until :meth:`close`.
    """

    root: Any

    def first_child(self, node: Any) -> Any: ...

    def last_child(self, node: Any) -> Any: ...

    def next_sibling(self, node: Any) -> Any: ...

    def previous_sibling(self, node: Any) -> Any: ...

    def focused(self) -> Any: ...

    def control_type(self, node: Any) -> int: ...

    def process_id(self, node: Any) -> int: ...

    def class_name(self, node: Any) -> str: ...

    def automation_id(self, node: Any) -> str: ...

    def is_keyboard_focusable(self, node: Any) -> bool: ...

    def is_offscreen(self, node: Any) -> bool: ...

    def rect(self, node: Any) -> Rect | None: ...

    def runtime_id(self, node: Any) -> tuple[int, ...] | None: ...

    def set_focus(self, node: Any) -> None: ...

    def close(self) -> None: ...


class UiaTreeClient(Protocol):
    """The UI Automation calls the provider makes (``platform.win_uia.UiAutomation``)."""

    def control_view(self, hwnd: int) -> ElementView: ...

    def foreground_window(self) -> int | None: ...

    def close(self) -> None: ...


class _Budget:
    """Counts the elements a walk steps to."""

    def __init__(self, limit: int) -> None:
        self.left = limit
        self.exhausted = False

    def take(self) -> bool:
        if self.left <= 0:
            self.exhausted = True
            return False
        self.left -= 1
        return True


def _children(
    view: ElementView, node: Any, budget: _Budget, *, backwards: bool = False
) -> Iterator[Any]:
    """The children of ``node`` in the control view, while the budget lasts."""
    if not budget.take():
        return
    child = view.last_child(node) if backwards else view.first_child(node)
    while child is not None:
        yield child
        if not budget.take():
            return
        child = view.previous_sibling(child) if backwards else view.next_sibling(child)


def find_documents(view: ElementView, root: Any, budget: _Budget) -> Iterator[Any]:
    """The ``RootWebArea`` documents below ``root``, in tree order, one at a time.

    A window can hold several (the Claude app has an empty one in front of the
    one with the sessions), so the caller walks on only while it has found no panes.
    """

    def visit(node: Any, depth: int) -> Iterator[Any]:
        for child in _children(view, node, budget):
            if view.control_type(child) == UIA_DOCUMENT_CONTROL_TYPE_ID:
                if view.automation_id(child) == ROOT_WEB_AREA:
                    yield child
                continue  # another document's web content is not ours to walk
            if depth < NATIVE_DEPTH:
                yield from visit(child, depth + 1)

    yield from visit(root, 1)


def find_panes(view: ElementView, document: Any, profile: AppProfile, budget: _Budget) -> list[Any]:
    """The pane groups below the web content ``document``, in tree order."""
    panes: list[Any] = []

    def visit(node: Any, depth: int) -> bool:
        here: list[Any] = []
        for child in _children(view, node, budget):
            if view.control_type(child) != UIA_GROUP_CONTROL_TYPE_ID:
                continue
            tokens = class_tokens(view.class_name(child))
            if profile.pane_token is not None and profile.pane_token in tokens:
                here.append(child)  # a pane: its contents are not walked
                continue
            if profile.is_container(tokens):
                # Its group children are the panes; their contents are not walked.
                panes.extend(
                    c
                    for c in _children(view, child, budget)
                    if view.control_type(c) == UIA_GROUP_CONTROL_TYPE_ID
                )
                return True
            if tokens & profile.skip_tokens or depth >= WEB_DEPTH:
                continue
            if visit(child, depth + 1):
                return True
        if here:
            panes.extend(here)
            return True
        return False

    visit(document, 1)
    return panes


def find_composer(view: ElementView, pane: Any, profile: AppProfile, budget: _Budget) -> Any:
    """The message box of ``pane``: the first ``Edit`` with the composer token
    seen from the bottom up, else the first keyboard-focusable ``Edit``."""
    fallback: list[Any] = []

    def visit(node: Any, depth: int) -> Any:
        for child in _children(view, node, budget, backwards=True):
            if view.control_type(child) == UIA_EDIT_CONTROL_TYPE_ID:
                if profile.composer_token in class_tokens(view.class_name(child)):
                    return child
                if not fallback and view.is_keyboard_focusable(child):
                    fallback.append(child)
                continue  # an editor's contents are text
            if depth < COMPOSER_DEPTH:
                found = visit(child, depth + 1)
                if found is not None:
                    return found
        return None

    found = visit(pane, 1)
    if found is not None:
        return found
    return fallback[0] if fallback else None


def pane_id(runtime_id: tuple[int, ...] | None, rect: Rect | None) -> Hashable:
    """A pane's id: its runtime id, else its (unclipped) rectangle."""
    if runtime_id:
        return runtime_id
    if rect is None:
        return None
    return ("rect", rect.x, rect.y, rect.w, rect.h)


def _within(inner: Rect, outer: Rect) -> bool:
    """Whether ``inner`` belongs to ``outer``: centre inside, and not larger."""
    return outer.contains(*inner.center) and inner.w <= outer.w and inner.h <= outer.h


@dataclass(frozen=True, slots=True)
class _Found:
    node: Any
    id: Hashable
    rect: Rect


class ChromiumAppProvider:
    """Side-by-side sessions of the desktop apps in :data:`PROFILES`."""

    name = PROVIDER_NAME

    def __init__(
        self,
        *,
        client_rect: Callable[[WindowRef], Rect | None],
        runner: Any = None,  # registry interface; this provider runs no commands
        client: UiaTreeClient | None = None,
        profiles: tuple[AppProfile, ...] = PROFILES,
    ) -> None:
        self._client_rect = client_rect
        self._client = client
        self._profiles = profiles
        #: The profile each window was detected with (``focus`` gets no app).
        self._window_profiles: dict[tuple[int, int | None], AppProfile] = {}

    def applies(self, app: AppIdentity) -> bool:
        return self._profile(app) is not None

    def detect(self, ref: WindowRef, app: AppIdentity) -> PaneSnapshot | None:
        hwnd = _hwnd(ref)
        profile = self._profile(app)
        if hwnd is None or profile is None:
            return None
        client = self._uia()
        try:
            if client.foreground_window() != hwnd:
                return None
            view = client.control_view(hwnd)
            try:
                found = self._panes(view, ref, profile)
                if len(found) < 2:
                    return None
                focus_rect = self._focused_rect(view, ref)
            finally:
                view.close()
        except OSError as exc:
            raise PaneError(f"UI Automation: {exc}") from exc
        self._window_profiles = {(hwnd, ref.pid): profile}
        panes = [
            Pane(f.id, f.rect, focus_rect is not None and _within(focus_rect, f.rect), self.name)
            for f in found
        ]
        panes.sort(key=lambda p: (p.rect.y, p.rect.x))
        return PaneSnapshot(ref.handle, tuple(panes), 0.0)

    def focus(self, ref: WindowRef, pane: Pane) -> bool:
        hwnd = _hwnd(ref)
        if hwnd is None or pane.provider != self.name or pane.id is None:
            return False
        profile = self._window_profiles.get((hwnd, ref.pid))
        if profile is None:
            return False
        client = self._uia()
        try:
            if client.foreground_window() != hwnd:
                return False  # SetFocus could bring a background window to the front
            view = client.control_view(hwnd)
            try:
                target = next((f for f in self._panes(view, ref, profile) if f.id == pane.id), None)
                if target is None:
                    return False
                composer = find_composer(view, target.node, profile, _Budget(COMPOSER_BUDGET))
                if composer is None:
                    log.debug("No message box found in the %s pane", profile.name)
                    return False
                if client.foreground_window() != hwnd:
                    return False  # another window came to the front during the walk
                view.set_focus(composer)
                return True
            finally:
                view.close()
        except OSError as exc:
            raise PaneError(f"UI Automation: {exc}") from exc

    # ------------------------------------------------------------- internals
    def _profile(self, app: AppIdentity) -> AppProfile | None:
        return next((p for p in self._profiles if p.matches(app)), None)

    def _panes(self, view: ElementView, ref: WindowRef, profile: AppProfile) -> list[_Found]:
        """The visible panes of the window (any number), in tree order."""
        root = view.root
        if root is None:
            return []
        budget = _Budget(NODE_BUDGET)
        nodes: list[Any] = []
        saw_document = False
        for document in find_documents(view, root, budget):
            saw_document = True
            nodes = find_panes(view, document, profile, budget)
            if nodes:
                break
        if not saw_document:
            log.debug("No web content in the %s window (yet)", profile.name)
            return []
        if budget.exhausted:
            log.debug("Walk of the %s window stopped after %d elements", profile.name, NODE_BUDGET)
        bounds = self._client_rect(ref) or ref.rect
        found: list[_Found] = []
        seen: set[Hashable] = set()
        for node in nodes:
            if view.is_offscreen(node):
                continue
            rect = view.rect(node)
            if rect is None:
                continue
            clipped = _clip(rect, bounds)
            if clipped is None or not profile.large_enough(clipped):
                continue
            ident = pane_id(view.runtime_id(node), rect)
            if ident is None or ident in seen:
                continue
            seen.add(ident)
            found.append(_Found(node, ident, clipped))
        return found

    def _focused_rect(self, view: ElementView, ref: WindowRef) -> Rect | None:
        """Rectangle of the element with the keyboard focus, if it is the app's."""
        element = view.focused()
        if element is None:
            return None
        expected = ref.pid
        if expected is None and view.root is not None:
            expected = view.process_id(view.root)
        if not expected or view.process_id(element) != expected:
            return None
        return view.rect(element)

    def close(self) -> None:
        """Release the UI Automation client (on the pane worker's thread)."""
        client, self._client = self._client, None
        self._window_profiles = {}
        if client is not None:
            client.close()

    def _uia(self) -> UiaTreeClient:
        """The UI Automation client, created on first use (on the worker thread)."""
        if self._client is None:
            self._client = UiAutomation()
        return self._client
