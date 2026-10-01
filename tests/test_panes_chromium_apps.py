"""Tests for the desktop-app pane provider (panes/providers/chromium_apps.py), its
place behind the deny-list (panes/registry.py) and the tree walk of the UI
Automation binding (platform/win_uia.py).

Everything runs against a fake control view shaped like the measured tree of
the Claude desktop app; no test asks a real window anything.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from eye_tracker.config import PaneSettings, Settings, describe_settings
from eye_tracker.panes.providers import chromium_apps
from eye_tracker.panes.providers.chromium_apps import (
    CHATGPT,
    CLAUDE,
    COMPOSER_BUDGET,
    NODE_BUDGET,
    PROVIDER_NAME,
    AppProfile,
    ChromiumAppProvider,
    class_tokens,
    profile_for,
)
from eye_tracker.panes.registry import PaneRegistry, default_providers, is_denied
from eye_tracker.panes.types import Pane, PaneError
from eye_tracker.panes.worker import PaneWorker
from eye_tracker.platform import win_uia
from eye_tracker.platform.win_uia import (
    UIA_DOCUMENT_CONTROL_TYPE_ID as DOCUMENT,
)
from eye_tracker.platform.win_uia import (
    UIA_EDIT_CONTROL_TYPE_ID as EDIT,
)
from eye_tracker.platform.win_uia import (
    UIA_GROUP_CONTROL_TYPE_ID as GROUP,
)
from eye_tracker.platform.win_uia import (
    UIA_PANE_CONTROL_TYPE_ID as NATIVE,
)
from eye_tracker.platform.win_uia import ComError, ControlView, UiaNode
from eye_tracker.types import AppIdentity, Rect, WindowRef
from panes_fakes import RecordingProvider, two_panes

WINDOW_TYPE = 50032  # UIA_WindowControlTypeId
TEXT = 50020  # UIA_TextControlTypeId
BUTTON = 50000  # UIA_ButtonControlTypeId

HWND = 0x5E0A1C
PID = 31337
WINDOW = WindowRef(handle=HWND, pid=PID, rect=Rect(1920, 0, 1920, 1040))
AREA = Rect(1920, 0, 1920, 1040)
APP = AppIdentity("claude", "Chrome_WidgetWin_1")

PRIMARY_RECT = Rect(2200, 36, 820, 1004)
SECONDARY_RECT = Rect(3020, 36, 820, 1004)
PRIMARY_ID = (42, 7, 101)
SECONDARY_ID = (42, 7, 102)
PRIMARY_CLASS = "dframe-pane dframe-pane-primary min-w-0 relative flex flex-col"
SECONDARY_CLASS = "dframe-pane dframe-pane-extra min-w-0 relative flex flex-col"


# ------------------------------------------------------------------ fake tree
@dataclass(eq=False)
class Node:
    """One element of the fake control view."""

    kind: int
    cls: str = ""
    aid: str = ""
    rect: Rect | None = None
    rid: tuple[int, ...] | None = None
    offscreen: bool = False
    focusable: bool = False
    pid: int = PID
    children: list[Node] = field(default_factory=list)
    parent: Node | None = None

    def add(self, *children: Node) -> Node:
        for child in children:
            child.parent = self
            self.children.append(child)
        return self


def walk(node: Node) -> Iterator[Node]:
    yield node
    for child in node.children:
        yield from walk(child)


@dataclass
class ClaudeTree:
    window: Node
    document: Node
    primary: Node
    secondary: Node
    primary_composer: Node
    secondary_composer: Node
    sidebar: Node


def conversation(messages: int) -> Node:
    """A transcript: one group per message, each with text inside."""
    transcript = Node(GROUP, "flex-1 overflow-y-auto")
    for i in range(messages):
        transcript.add(
            Node(GROUP, "font-claude-message").add(
                Node(TEXT), Node(GROUP, f"m-{i}").add(Node(TEXT))
            )
        )
    return transcript


def session(cls: str, rect: Rect, rid: tuple[int, ...], *, messages: int = 3) -> tuple[Node, Node]:
    composer = Node(
        EDIT,
        "tiptap ProseMirror break-words",
        rect=Rect(rect.x + 53, 975, 688, 20),
        focusable=True,
    )
    pane = Node(GROUP, cls, rect=rect, rid=rid).add(
        Node(GROUP, "sticky top-0").add(Node(BUTTON)),
        conversation(messages),
        Node(GROUP, "composer-area").add(
            Node(GROUP, "relative").add(composer), Node(GROUP, "toolbar").add(Node(BUTTON))
        ),
    )
    return pane, composer


def claude_tree(
    *,
    built: bool = True,
    secondary: bool = True,
    messages: int = 3,
    sidebar_chats: int = 50,
) -> ClaudeTree:
    """The Claude window as measured (native views down to the web content)."""
    primary, primary_composer = session(PRIMARY_CLASS, PRIMARY_RECT, PRIMARY_ID, messages=messages)
    extra, extra_composer = session(
        SECONDARY_CLASS, SECONDARY_RECT, SECONDARY_ID, messages=messages
    )
    sidebar = Node(GROUP, "dframe-sidebar df-hub-rail", rect=Rect(1920, 0, 280, 1040))
    for _ in range(sidebar_chats):
        sidebar.add(Node(GROUP, "chat-item").add(Node(TEXT)))
    content = Node(GROUP, "dframe-content flex").add(primary)
    if secondary:
        content.add(extra)
    document = Node(DOCUMENT, aid="RootWebArea", rect=AREA, focusable=True)
    if built:
        document.add(
            Node(GROUP, "bg-surface-1 text-primary font-sans min-h-screen").add(
                Node(GROUP, aid="root").add(sidebar, content)
            )
        )
    innermost = Node(NATIVE, "View").add(document)
    views = innermost
    for _ in range(3):
        views = Node(NATIVE, "View").add(views)
    window = Node(WINDOW_TYPE, "Chrome_WidgetWin_1", rect=AREA).add(
        Node(NATIVE, "RootView").add(
            Node(NATIVE, "NonClientView").add(
                Node(NATIVE, "WinFrameView").add(
                    Node(NATIVE, "TitleBar").add(Node(BUTTON), Node(BUTTON), Node(BUTTON)),
                    Node(NATIVE, "ClientView").add(views),
                )
            )
        )
    )
    return ClaudeTree(window, document, primary, extra, primary_composer, extra_composer, sidebar)


class FakeView:
    """A control view over a fake tree; counts steps, records property reads."""

    def __init__(self, client: FakeClient, root: Node) -> None:
        self._client = client
        self.root: Node | None = root
        self.steps = 0
        self.closed = False
        self.reads: list[tuple[str, Node]] = []

    # navigation
    def _step(self, node: Node | None) -> Node | None:
        assert not self.closed
        self.steps += 1
        self._client.maybe_fail()
        return node

    def first_child(self, node: Node) -> Node | None:
        return self._step(node.children[0] if node.children else None)

    def last_child(self, node: Node) -> Node | None:
        return self._step(node.children[-1] if node.children else None)

    def next_sibling(self, node: Node) -> Node | None:
        siblings = node.parent.children if node.parent else [node]
        i = siblings.index(node)
        return self._step(siblings[i + 1] if i + 1 < len(siblings) else None)

    def previous_sibling(self, node: Node) -> Node | None:
        siblings = node.parent.children if node.parent else [node]
        i = siblings.index(node)
        return self._step(siblings[i - 1] if i > 0 else None)

    def focused(self) -> Node | None:
        assert not self.closed
        return self._client.focused

    # properties
    def _read(self, what: str, node: Node) -> None:
        assert not self.closed
        self.reads.append((what, node))

    def control_type(self, node: Node) -> int:
        self._read("control_type", node)
        return node.kind

    def process_id(self, node: Node) -> int:
        self._read("process_id", node)
        return node.pid

    def class_name(self, node: Node) -> str:
        self._read("class_name", node)
        return node.cls

    def automation_id(self, node: Node) -> str:
        self._read("automation_id", node)
        return node.aid

    def is_keyboard_focusable(self, node: Node) -> bool:
        self._read("is_keyboard_focusable", node)
        return node.focusable

    def is_offscreen(self, node: Node) -> bool:
        self._read("is_offscreen", node)
        return node.offscreen

    def rect(self, node: Node) -> Rect | None:
        self._read("rect", node)
        return node.rect

    def runtime_id(self, node: Node) -> tuple[int, ...] | None:
        self._read("runtime_id", node)
        return node.rid

    def set_focus(self, node: Node) -> None:
        assert not self.closed
        self._client.focus_set.append(node)
        self._client.focused = node

    def close(self) -> None:
        self.closed = True


class FakeClient:
    """Answers like ``win_uia.UiAutomation``; records every view it opened."""

    def __init__(
        self,
        tree: ClaudeTree | None = None,
        *,
        foreground: int | None = HWND,
        focused: Node | None = None,
    ) -> None:
        self.tree = tree if tree is not None else claude_tree()
        self.foreground = foreground
        self.focused = focused if focused is not None else self.tree.primary_composer
        self.views: list[FakeView] = []
        self.calls: list[str] = []
        self.focus_set: list[Node] = []
        self.fail: BaseException | None = None

    def maybe_fail(self) -> None:
        if self.fail is not None:
            raise self.fail

    def control_view(self, hwnd: int) -> FakeView:
        self.calls.append("control_view")
        assert hwnd == HWND
        view = FakeView(self, self.tree.window)
        self.views.append(view)
        return view

    def foreground_window(self) -> int | None:
        self.calls.append("foreground")
        return self.foreground

    @property
    def all_closed(self) -> bool:
        return all(v.closed for v in self.views)


GPT_APP = AppIdentity("chatgpt", "Chrome_WidgetWin_1")
MAIN_RECT = Rect(2261, 96, 884, 940)
SIDE_RECT = Rect(3145, 96, 691, 940)
MAIN_ID = (42, 9, 1)
SIDE_ID = (42, 9, 2)


def chatgpt_tree(
    *, side: bool = True, overlay: bool = True, build_hash: str = "bo1ta_2", messages: int = 3
) -> ClaudeTree:
    """The ChatGPT window as measured: main conversation and side chat."""
    main_composer = Node(EDIT, "ProseMirror", rect=Rect(2347, 936, 712, 44), focusable=True)
    main = Node(GROUP, "relative flex h-full flex-col min-h-0", rect=MAIN_RECT, rid=MAIN_ID).add(
        Node(GROUP, "group/thread-scroll-layout has-[[data-composer]]").add(
            conversation(messages), Node(GROUP, "composer").add(main_composer)
        )
    )
    side_composer = Node(
        EDIT, "ProseMirror ProseMirror-focused", rect=Rect(3174, 936, 634, 44), focusable=True
    )
    side_pane = Node(
        GROUP, "relative z-[41] h-full min-h-0 min-w-0 shrink-0", rect=SIDE_RECT, rid=SIDE_ID
    ).add(
        Node(GROUP, "min-h-0 flex-1").add(
            Node(
                NATIVE,
                "min-h-0 min-w-0 flex-1 outline-none relative",
                aid="app-shell-tab-panel-app-shell-tab:4",
                rect=Rect(3146, 96, 690, 940),
            ).add(conversation(messages), side_composer)
        )
    )
    container = Node(
        GROUP,
        f"outline-none _MainContentSurface_{build_hash}",
        aid="_r_bs_",
        rect=Rect(2260, 44, 1576, 992),
    ).add(main)
    if side:
        container.add(side_pane)
    if overlay:
        container.add(Node(GROUP, "absolute toast", rect=Rect(3000, 100, 180, 40), rid=(42, 9, 3)))
    sidebar = Node(GROUP, "app-shell-left-panel pointer-events-auto", rect=Rect(1920, 44, 340, 992))
    for _ in range(100):
        sidebar.add(Node(GROUP, "chat-item").add(Node(TEXT)))
    document = Node(DOCUMENT, aid="RootWebArea", rect=AREA, focusable=True).add(
        Node(GROUP, "").add(Node(GROUP, aid="root").add(sidebar, container))
    )
    window = Node(WINDOW_TYPE, "Chrome_WidgetWin_1", rect=AREA).add(
        Node(NATIVE, "RootView").add(
            Node(NATIVE, "NonClientView").add(
                Node(NATIVE, "ChromeNodeFrameView").add(
                    Node(NATIVE, "ChromeNodeClientView").add(
                        Node(NATIVE, "View").add(Node(NATIVE, "Chrome_WidgetWin_1").add(document))
                    )
                )
            )
        )
    )
    return ClaudeTree(window, document, main, side_pane, main_composer, side_composer, sidebar)


def provider(client: FakeClient) -> ChromiumAppProvider:
    return ChromiumAppProvider(
        client_rect=lambda ref: AREA if ref.handle == HWND else None, client=client
    )


# ---------------------------------------------------------------- profiles
def test_only_the_claude_and_chatgpt_apps_have_a_profile() -> None:
    assert chromium_apps.PROFILES == (CLAUDE, CHATGPT)
    assert profile_for(APP) is CLAUDE
    assert profile_for(AppIdentity("Claude", "Chrome_WidgetWin_1")) is CLAUDE
    assert profile_for(GPT_APP) is CHATGPT
    assert profile_for(AppIdentity("ChatGPT", "Chrome_WidgetWin_1")) is CHATGPT
    # Process name and window class must both match.
    assert profile_for(AppIdentity("claude", "Chrome_WidgetWin_0")) is None
    assert profile_for(AppIdentity("claude", "com.anthropic.claudefordesktop")) is None
    assert profile_for(AppIdentity("chatgpt", "com.openai.chat")) is None
    assert profile_for(AppIdentity("code", "Chrome_WidgetWin_1")) is None
    assert profile_for(None) is None


def test_a_profile_finds_panes_one_way() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        AppProfile("x", frozenset({"x"}), "c", "ProseMirror")
    with pytest.raises(ValueError, match="exactly one"):
        AppProfile("x", frozenset({"x"}), "c", "ProseMirror", pane_token="a", container_prefix="b")


def test_the_container_is_matched_by_token_prefix() -> None:
    assert CHATGPT.is_container(class_tokens("outline-none _MainContentSurface_bo1ta_2"))
    assert CHATGPT.is_container(class_tokens("_MainContentSurface_zz9_7"))  # another build
    assert not CHATGPT.is_container(class_tokens("x_MainContentSurface_bo1ta_2"))
    assert not CHATGPT.is_container(class_tokens("MainContentSurface outline-none"))
    assert not CLAUDE.is_container(class_tokens("_MainContentSurface_bo1ta_2"))


def test_creating_the_provider_touches_nothing() -> None:
    p = ChromiumAppProvider(client_rect=lambda ref: None, runner=None)
    assert p.name == PROVIDER_NAME == "desktop_apps"
    assert p.applies(APP)
    assert not p.applies(AppIdentity("cursor", "Chrome_WidgetWin_1"))
    assert p._client is None


def test_class_tokens_are_whole_words() -> None:
    assert class_tokens(PRIMARY_CLASS) >= {"dframe-pane", "dframe-pane-primary"}
    assert "dframe-pane" not in class_tokens("dframe-pane-primary min-w-0")
    assert "dframe-pane" not in class_tokens("xdframe-pane")
    assert class_tokens("  a\tb\n c ") == {"a", "b", "c"}
    assert class_tokens("") == frozenset()


# ------------------------------------------------------------------- detect
def test_detects_the_two_sessions_and_the_focused_one() -> None:
    client = FakeClient()
    snapshot = provider(client).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.window_handle == HWND
    assert [p.rect for p in snapshot.panes] == [PRIMARY_RECT, SECONDARY_RECT]
    assert [p.id for p in snapshot.panes] == [PRIMARY_ID, SECONDARY_ID]
    assert {p.provider for p in snapshot.panes} == {"desktop_apps"}
    assert snapshot.focused is not None
    assert snapshot.focused.id == PRIMARY_ID
    assert client.calls == ["foreground", "control_view"]
    assert client.all_closed

    client.focused = client.tree.secondary_composer
    snapshot = provider(client).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.focused is not None
    assert snapshot.focused.id == SECONDARY_ID


def test_the_walk_is_short_and_reads_no_more_than_it_needs() -> None:
    client = FakeClient(claude_tree(messages=200, sidebar_chats=300))
    assert provider(client).detect(WINDOW, APP) is not None
    [view] = client.views
    # Native views, the document, three groups down to the panes: a few dozen steps.
    assert view.steps <= 40, view.steps
    tree = client.tree
    # Of the focused element only its process and rectangle are read.
    assert {w for w, n in view.reads if n is tree.primary_composer} == {"process_id", "rect"}
    touched = {id(node) for _, node in view.reads if node is not tree.primary_composer}
    inside = [n for pane in (tree.primary, tree.secondary) for n in walk(pane)][1:]
    inside = [n for n in inside if n not in (tree.primary, tree.secondary)]
    assert not any(id(n) in touched for n in inside)  # pane contents are not walked
    assert not any(id(n) in touched for n in list(walk(tree.sidebar))[1:])  # nor the sidebar
    kinds = {what for what, _ in view.reads}
    assert kinds <= {
        "control_type",
        "class_name",
        "automation_id",
        "is_offscreen",
        "rect",
        "runtime_id",
        "process_id",
    }
    # Class names only of groups, automation ids only of documents.
    assert all(n.kind == GROUP for what, n in view.reads if what == "class_name")
    assert all(n.kind == DOCUMENT for what, n in view.reads if what == "automation_id")


def test_one_session_is_not_a_split() -> None:
    assert provider(FakeClient(claude_tree(secondary=False))).detect(WINDOW, APP) is None


def test_hidden_and_empty_sessions_do_not_count() -> None:
    tree = claude_tree()
    tree.secondary.offscreen = True
    assert provider(FakeClient(tree)).detect(WINDOW, APP) is None
    tree = claude_tree()
    tree.secondary.rect = None
    assert provider(FakeClient(tree)).detect(WINDOW, APP) is None
    tree = claude_tree()
    tree.secondary.rect = Rect(5000, 36, 820, 1004)  # outside the window
    assert provider(FakeClient(tree)).detect(WINDOW, APP) is None


def test_only_whole_class_tokens_make_a_pane() -> None:
    tree = claude_tree()
    tree.secondary.cls = "dframe-pane-extra min-w-0"  # no "dframe-pane" token
    assert provider(FakeClient(tree)).detect(WINDOW, APP) is None
    tree.secondary.cls = "xdframe-pane"
    assert provider(FakeClient(tree)).detect(WINDOW, APP) is None
    tree.secondary.cls = "min-w-0 dframe-pane"  # the token anywhere in the list
    assert provider(FakeClient(tree)).detect(WINDOW, APP) is not None


def test_a_pane_that_is_not_a_group_does_not_count() -> None:
    tree = claude_tree()
    tree.secondary.kind = NATIVE
    assert provider(FakeClient(tree)).detect(WINDOW, APP) is None


def test_no_pane_is_focused_unless_the_focus_is_inside_one() -> None:
    tree = claude_tree()
    # The whole page has the focus: larger than a pane, so no pane has it.
    snapshot = provider(FakeClient(tree, focused=tree.document)).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.focused is None
    # The sidebar has it.
    snapshot = provider(FakeClient(tree, focused=tree.sidebar)).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.focused is None
    # The pane itself has it.
    snapshot = provider(FakeClient(tree, focused=tree.secondary)).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.focused is not None
    assert snapshot.focused.id == SECONDARY_ID
    # An element of another process at the same place does not count.
    stranger = Node(EDIT, rect=Rect(3073, 975, 688, 20), pid=PID + 1)
    snapshot = provider(FakeClient(tree, focused=stranger)).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.focused is None


def test_without_a_known_pid_the_windows_process_is_compared() -> None:
    tree = claude_tree()
    ref = WindowRef(handle=HWND, pid=None, rect=AREA)
    snapshot = provider(FakeClient(tree)).detect(ref, APP)
    assert snapshot is not None
    assert snapshot.focused is not None
    assert snapshot.focused.id == PRIMARY_ID
    tree.window.pid = PID + 1
    snapshot = provider(FakeClient(tree)).detect(ref, APP)
    assert snapshot is not None
    assert snapshot.focused is None


def test_without_runtime_ids_the_rectangles_are_the_ids() -> None:
    tree = claude_tree()
    tree.primary.rid = None
    tree.secondary.rid = ()
    snapshot = provider(FakeClient(tree)).detect(WINDOW, APP)
    assert snapshot is not None
    assert [p.id for p in snapshot.panes] == [
        ("rect", 2200, 36, 820, 1004),
        ("rect", 3020, 36, 820, 1004),
    ]


def test_a_tree_not_built_yet_is_no_panes_not_a_failure(qapp: Any) -> None:
    # Chromium builds its accessibility tree once asked: first only native views,
    # then web content without panes, then everything.
    client = FakeClient(claude_tree(built=False))
    client.tree.document.kind = NATIVE  # not even the document yet
    worker = PaneWorker(PaneRegistry([provider(client)], desktop_apps=True), synchronous=True)
    detected: list[Any] = []
    worker.detected.connect(detected.append)
    for _ in range(5):
        worker.request_detect(WINDOW, APP)
    client.tree = claude_tree(built=False)
    for _ in range(5):
        worker.request_detect(WINDOW, APP)
    assert all(r.snapshot is None for r in detected)
    assert worker.disabled_providers == frozenset()
    assert worker._failures == {}
    client.tree = claude_tree()
    worker.request_detect(WINDOW, APP)
    assert detected[-1].snapshot is not None
    assert detected[-1].snapshot.provider == "desktop_apps"
    assert client.all_closed


def test_another_document_is_never_entered() -> None:
    tree = claude_tree()
    other = Node(DOCUMENT, aid="DevTools").add(
        Node(GROUP, "x").add(Node(GROUP, "dframe-pane", rect=PRIMARY_RECT, rid=(9,)))
    )
    tree.window.children[0].children.insert(0, other)
    other.parent = tree.window.children[0]
    client = FakeClient(tree)
    snapshot = provider(client).detect(WINDOW, APP)
    assert snapshot is not None
    assert [p.id for p in snapshot.panes] == [PRIMARY_ID, SECONDARY_ID]
    touched = {id(n) for _, n in client.views[0].reads}
    assert not any(id(n) in touched for n in list(walk(other))[1:])


def test_the_walk_stops_at_its_budget() -> None:
    tree = claude_tree()
    # Thousands of groups before the content, without a pane among them.
    wide = Node(GROUP, "virtual-list")
    for _ in range(5000):
        wide.add(Node(GROUP, "row").add(Node(GROUP, "cell")))
    root_group = tree.document.children[0].children[0]
    root_group.children.insert(0, wide)
    wide.parent = root_group
    client = FakeClient(tree)
    assert provider(client).detect(WINDOW, APP) is None
    assert client.views[0].steps <= NODE_BUDGET
    assert client.all_closed

    # A deep chain of groups: at most WEB_DEPTH levels below the document.
    tree = claude_tree()
    deep = tree.document
    for _ in range(50):
        child = Node(GROUP, "nest")
        deep.children.insert(0, child)
        child.parent = deep
        deep = child
    deep.add(Node(GROUP, "dframe-pane", rect=PRIMARY_RECT, rid=(1,)))
    client = FakeClient(tree)
    provider(client).detect(WINDOW, APP)
    reached = {id(n) for _, n in client.views[0].reads}
    assert id(deep) not in reached


def test_a_huge_native_tree_stops_at_the_budget_too() -> None:
    window = Node(WINDOW_TYPE)
    for _ in range(2000):
        window.add(Node(NATIVE, "View").add(Node(NATIVE, "View")))
    tree = claude_tree()
    tree.window = window
    client = FakeClient(tree)
    assert provider(client).detect(WINDOW, APP) is None
    assert client.views[0].steps <= NODE_BUDGET


def test_nothing_is_asked_unless_the_window_is_in_the_foreground() -> None:
    client = FakeClient(foreground=0x1234)
    p = provider(client)
    assert p.detect(WINDOW, APP) is None
    assert p.focus(WINDOW, Pane(SECONDARY_ID, SECONDARY_RECT, False, PROVIDER_NAME)) is False
    assert client.calls == ["foreground"]  # focus() has not seen this window yet
    assert p.detect(WindowRef(handle=None), APP) is None
    assert p.detect(WINDOW, AppIdentity("code", "Chrome_WidgetWin_1")) is None
    assert client.views == []


def test_com_errors_become_pane_errors_and_release_the_view() -> None:
    client = FakeClient()
    p = provider(client)
    assert p.detect(WINDOW, APP) is not None
    client.fail = ComError("IUIAutomationTreeWalker::GetFirstChildElement", 0x80040201)
    with pytest.raises(PaneError, match="0x80040201"):
        p.detect(WINDOW, APP)
    with pytest.raises(PaneError):
        p.focus(WINDOW, Pane(SECONDARY_ID, SECONDARY_RECT, False, PROVIDER_NAME))
    assert client.all_closed


# -------------------------------------------------------------------- focus
def _detected(client: FakeClient) -> tuple[ChromiumAppProvider, Any]:
    p = provider(client)
    snapshot = p.detect(WINDOW, APP)
    assert snapshot is not None
    return p, snapshot


def test_focus_goes_to_the_sessions_message_box() -> None:
    client = FakeClient()
    p, snapshot = _detected(client)
    assert p.focus(WINDOW, snapshot.panes[1]) is True
    assert client.focus_set == [client.tree.secondary_composer]
    assert client.all_closed
    # And the next detection sees that session focused.
    again = p.detect(WINDOW, APP)
    assert again is not None
    assert again.focused is not None
    assert again.focused.id == SECONDARY_ID


def test_focus_finds_the_message_box_after_a_long_conversation() -> None:
    client = FakeClient(claude_tree(messages=2000))
    p, snapshot = _detected(client)
    assert p.focus(WINDOW, snapshot.panes[1]) is True
    assert client.focus_set == [client.tree.secondary_composer]
    assert client.views[-1].steps <= NODE_BUDGET + COMPOSER_BUDGET


def test_focus_prefers_the_prosemirror_editor() -> None:
    tree = claude_tree()
    toolbar = tree.secondary.children[-1].children[-1]
    search = Node(EDIT, "search-input", rect=Rect(3500, 990, 100, 20), focusable=True)
    toolbar.add(search)  # walked before the message box (bottom up)
    client = FakeClient(tree)
    p, snapshot = _detected(client)
    assert p.focus(WINDOW, snapshot.panes[1]) is True
    assert client.focus_set == [tree.secondary_composer]


def test_focus_falls_back_to_a_focusable_editor() -> None:
    tree = claude_tree()
    tree.secondary_composer.cls = "some-textarea"
    client = FakeClient(tree)
    p, snapshot = _detected(client)
    assert p.focus(WINDOW, snapshot.panes[1]) is True
    assert client.focus_set == [tree.secondary_composer]
    # Not focusable either: nothing to focus.
    tree.secondary_composer.focusable = False
    assert p.focus(WINDOW, snapshot.panes[1]) is False
    assert client.focus_set == [tree.secondary_composer]  # unchanged
    assert client.all_closed


def test_focus_of_a_session_that_is_gone_fails() -> None:
    client = FakeClient()
    p, snapshot = _detected(client)
    assert p.focus(WINDOW, Pane((42, 7, 999), SECONDARY_RECT, False, PROVIDER_NAME)) is False
    assert p.focus(WINDOW, Pane(SECONDARY_ID, SECONDARY_RECT, False, "windows_terminal")) is False
    client.tree.secondary.offscreen = True
    assert p.focus(WINDOW, snapshot.panes[1]) is False
    assert client.focus_set == []
    # A window never detected is not focused into.
    other = provider(FakeClient())
    assert other.focus(WINDOW, snapshot.panes[1]) is False


# ---------------------------------------------------------------- ChatGPT
def test_chatgpt_main_conversation_and_side_chat() -> None:
    tree = chatgpt_tree(messages=100)
    client = FakeClient(tree, focused=tree.secondary_composer)
    snapshot = provider(client).detect(WINDOW, GPT_APP)
    assert snapshot is not None
    assert [p.id for p in snapshot.panes] == [MAIN_ID, SIDE_ID]  # not the small overlay
    assert [p.rect for p in snapshot.panes] == [MAIN_RECT, SIDE_RECT]
    assert snapshot.focused is not None
    assert snapshot.focused.id == SIDE_ID
    [view] = client.views
    assert view.steps <= 40, view.steps
    touched = {id(n) for _, n in view.reads if n is not tree.secondary_composer}
    assert not any(id(n) in touched for n in list(walk(tree.sidebar))[1:])
    for pane in (tree.primary, tree.secondary):
        assert not any(id(n) in touched for n in list(walk(pane))[1:])
    assert client.all_closed


def test_chatgpt_container_of_another_build_is_found() -> None:
    tree = chatgpt_tree(build_hash="x7k2p_9")
    snapshot = provider(FakeClient(tree)).detect(WINDOW, GPT_APP)
    assert snapshot is not None
    assert snapshot.focused is not None
    assert snapshot.focused.id == MAIN_ID


def test_chatgpt_without_a_side_chat_is_not_a_split() -> None:
    # The main conversation and a small overlay: one pane.
    assert provider(FakeClient(chatgpt_tree(side=False))).detect(WINDOW, GPT_APP) is None


def test_chatgpt_container_without_prefix_token_is_not_used() -> None:
    tree = chatgpt_tree()
    container = tree.primary.parent
    assert container is not None
    container.cls = "outline-none MainContentSurface"
    assert provider(FakeClient(tree)).detect(WINDOW, GPT_APP) is None


def test_chatgpt_focus_goes_to_the_message_box_of_each_side() -> None:
    tree = chatgpt_tree(messages=500)
    client = FakeClient(tree, focused=tree.secondary_composer)
    p = provider(client)
    snapshot = p.detect(WINDOW, GPT_APP)
    assert snapshot is not None
    assert p.focus(WINDOW, snapshot.panes[0]) is True
    assert client.focus_set == [tree.primary_composer]
    assert p.focus(WINDOW, snapshot.panes[1]) is True
    assert client.focus_set == [tree.primary_composer, tree.secondary_composer]
    assert client.all_closed


def test_the_tree_not_built_yet_is_no_panes_for_chatgpt_too() -> None:
    tree = chatgpt_tree()
    tree.document.children.clear()
    assert provider(FakeClient(tree)).detect(WINDOW, GPT_APP) is None


# --------------------------------------------------------------- deny-list
DENIED_ALWAYS = [
    AppIdentity("code", "Chrome_WidgetWin_1"),
    AppIdentity("code - insiders", "Chrome_WidgetWin_1"),
    AppIdentity("cursor", "Chrome_WidgetWin_1"),
    AppIdentity("antigravity", "Chrome_WidgetWin_1"),
    AppIdentity("windsurf", "Chrome_WidgetWin_1"),
    # The ChatGPT process, but not its main window class.
    AppIdentity("chatgpt", "Chrome_WidgetWin_0"),
    AppIdentity("chrome", "Chrome_WidgetWin_1"),
    AppIdentity("msedge", "Chrome_WidgetWin_1"),
    AppIdentity("someapp", "Chrome_WidgetWin_1"),
    # The Claude process, but not its main window class.
    AppIdentity("claude", "Chrome_WidgetWin_0"),
    AppIdentity("claude", ""),
    # Another process with the Claude window class is not Claude.
    AppIdentity("claude-helper", "Chrome_WidgetWin_1"),
]


def _registry(client: FakeClient, other: RecordingProvider, *, desktop_apps: bool) -> PaneRegistry:
    return PaneRegistry([other, provider(client)], desktop_apps=desktop_apps)


PROFILED = [(APP, claude_tree), (GPT_APP, chatgpt_tree)]


@pytest.mark.parametrize(("app", "tree"), PROFILED, ids=["claude", "chatgpt"])
def test_profiled_apps_are_never_asked_while_the_setting_is_off(
    qapp: Any, app: AppIdentity, tree: Any
) -> None:
    client = FakeClient(tree())
    other = RecordingProvider("tmux", snapshot=lambda ref: two_panes(ref.handle))
    registry = _registry(client, other, desktop_apps=False)
    assert is_denied(app)
    assert registry.providers_for(app) == []
    worker = PaneWorker(registry, synchronous=True)
    results: list[Any] = []
    worker.detected.connect(results.append)
    worker.request_detect(WINDOW, app)
    assert client.calls == []
    assert other.calls == []
    assert results[0].snapshot is None


@pytest.mark.parametrize(("app", "tree"), PROFILED, ids=["claude", "chatgpt"])
def test_with_the_setting_on_profiled_apps_go_to_this_provider_alone(
    qapp: Any, app: AppIdentity, tree: Any
) -> None:
    client = FakeClient(tree())
    other = RecordingProvider("tmux", snapshot=lambda ref: two_panes(ref.handle))
    registry = _registry(client, other, desktop_apps=True)
    assert not is_denied(app, desktop_apps=True)
    assert [p.name for p in registry.providers_for(app)] == ["desktop_apps"]
    worker = PaneWorker(registry, synchronous=True)
    results: list[Any] = []
    worker.detected.connect(results.append)
    worker.request_detect(WINDOW, app)
    assert client.calls == ["foreground", "control_view"]
    assert other.calls == []  # tmux is not asked about the app's window
    assert results[0].snapshot is not None
    assert results[0].snapshot.provider == "desktop_apps"
    # Terminals still go to their providers, never to this one.
    terminal = AppIdentity("windowsterminal", "CASCADIA_HOSTING_WINDOW_CLASS")
    assert [p.name for p in registry.providers_for(terminal)] == ["tmux"]


@pytest.mark.parametrize("desktop_apps", [False, True])
@pytest.mark.parametrize("app", DENIED_ALWAYS, ids=lambda a: f"{a.process}|{a.app_id}")
def test_other_chromium_apps_stay_denied(qapp: Any, app: AppIdentity, desktop_apps: bool) -> None:
    client = FakeClient()
    other = RecordingProvider("tmux", snapshot=lambda ref: two_panes(ref.handle))
    registry = _registry(client, other, desktop_apps=desktop_apps)
    assert is_denied(app, desktop_apps=desktop_apps)
    assert registry.providers_for(app) == []
    worker = PaneWorker(registry, synchronous=True)
    results: list[Any] = []
    worker.detected.connect(results.append)
    worker.request_detect(WINDOW, app)
    assert client.calls == []
    assert other.calls == []
    assert results[0].snapshot is None


def test_default_providers_offer_desktop_apps_only_when_on_and_on_windows() -> None:
    settings = PaneSettings()
    assert settings.desktop_apps is False

    def names(system: str) -> list[str]:
        found = default_providers(settings, client_rect=lambda ref: None, system=system)
        return [p.name for p in found]

    assert "desktop_apps" not in names("win32")
    settings.desktop_apps = True
    assert names("win32") == ["wezterm", "windows_terminal", "desktop_apps", "tmux"]
    assert "desktop_apps" not in names("linux")
    assert "desktop_apps" not in names("darwin")


def test_the_setting_is_off_by_default_and_says_what_it_costs() -> None:
    assert Settings().panes.desktop_apps is False
    doc = next(row["doc"] for row in describe_settings() if row["key"] == "panes.desktop_apps")
    assert "Claude" in doc
    assert "ChatGPT" in doc
    assert "Codex" in doc
    assert "accessibility tree" in doc
    assert "CPU" in doc


# ------------------------------------------------------------------ binding
def test_the_vtable_slots_match_uiautomationclient_h() -> None:
    # Counted from the C ...Vtbl structs of UIAutomationClient.h (SDK 10.0.26100).
    assert win_uia._UIA_GET_FOCUSED_ELEMENT == 8
    assert win_uia._UIA_GET_CONTROL_VIEW_WALKER == 14
    assert (
        win_uia._WALKER_FIRST_CHILD,
        win_uia._WALKER_LAST_CHILD,
        win_uia._WALKER_NEXT_SIBLING,
        win_uia._WALKER_PREVIOUS_SIBLING,
    ) == (4, 5, 6, 7)
    assert win_uia._EL_PROCESS_ID == 20
    assert win_uia._EL_CONTROL_TYPE == 21
    assert win_uia._EL_IS_KEYBOARD_FOCUSABLE == 27
    assert win_uia._EL_AUTOMATION_ID == 29
    assert win_uia._EL_CLASS_NAME == 30
    assert (EDIT, GROUP, DOCUMENT, NATIVE) == (50004, 50026, 50030, 50033)


UIA = 0xA  # the IUIAutomation pointer of the fake


class _FakeCom:
    """Stands in for the vtable calls of ``ControlView`` (pointers are ints)."""

    def __init__(self, children: dict[int, list[int]], *, fail_at: int | None = None) -> None:
        self.children = children
        self.next_ptr = 1000
        self.handed: list[int] = []
        self.released: list[int] = []
        self.fail_at = fail_at

    def out_ptr(self, ptr: int, index: int, what: str, *args: Any, argtypes: Any = ()) -> int:
        if self.fail_at is not None and index == self.fail_at:
            raise ComError(what, 0x80004005)
        if ptr == UIA:  # IUIAutomation (its slots overlap the walker's)
            if index == win_uia._UIA_GET_CONTROL_VIEW_WALKER:
                return self._new()
            if index == win_uia._UIA_ELEMENT_FROM_HANDLE:
                return self._new(1)
            if index == win_uia._UIA_GET_FOCUSED_ELEMENT:
                return self._new(99)
            raise AssertionError(index)
        target = args[0].value if args else 0
        if index == win_uia._WALKER_FIRST_CHILD:
            kids = self.children.get(target, [])
            return self._new(kids[0]) if kids else 0
        if index == win_uia._WALKER_NEXT_SIBLING:
            for kids in self.children.values():
                if target in kids and kids.index(target) + 1 < len(kids):
                    return self._new(kids[kids.index(target) + 1])
            return 0
        raise AssertionError(index)

    def _new(self, ident: int | None = None) -> int:
        ptr = ident if ident is not None else self.next_ptr
        self.next_ptr += 1
        self.handed.append(ptr)
        return ptr

    def release(self, ptr: int | None) -> None:
        if ptr:
            self.released.append(ptr)


def test_control_view_releases_everything_it_handed_out(monkeypatch: pytest.MonkeyPatch) -> None:
    com = _FakeCom({1: [2, 3], 2: [4]})
    monkeypatch.setattr(win_uia, "_out_ptr", com.out_ptr)
    monkeypatch.setattr(win_uia, "_release", com.release)
    with ControlView(object(), UIA, HWND) as view:  # type: ignore[arg-type]
        assert view.root == UiaNode(1)
        first = view.first_child(UiaNode(1))
        assert first == UiaNode(2)
        assert view.first_child(UiaNode(2)) == UiaNode(4)
        assert view.next_sibling(UiaNode(2)) == UiaNode(3)
        assert view.next_sibling(UiaNode(3)) is None
        assert view.first_child(UiaNode(4)) is None
        assert view.focused() == UiaNode(99)
        assert view.held == 5
    assert sorted(com.released) == sorted(com.handed)
    with pytest.raises(ComError):
        view.first_child(UiaNode(1))  # closed
    view.close()  # idempotent
    assert sorted(com.released) == sorted(com.handed)


def test_control_view_releases_the_walker_when_creating_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    com = _FakeCom({}, fail_at=win_uia._UIA_ELEMENT_FROM_HANDLE)
    monkeypatch.setattr(win_uia, "_out_ptr", com.out_ptr)
    monkeypatch.setattr(win_uia, "_release", com.release)
    with pytest.raises(ComError):
        ControlView(object(), UIA, HWND)  # type: ignore[arg-type]
    assert com.handed == com.released != []


def test_control_view_refuses_other_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(OSError, match="only available on Windows"):
        win_uia.UiAutomation().control_view(HWND)


@pytest.mark.skipif(sys.platform != "win32", reason="needs oleaut32")
def test_smoke_reads_and_frees_a_bstr(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    api = win_uia._api()
    freed: list[int] = []

    class Api:
        SysStringLen = api.SysStringLen

        @staticmethod
        def SysFreeString(bstr: int) -> None:
            freed.append(bstr)
            api.SysFreeString(bstr)

    allocated: list[int] = []

    def method(ptr: int, index: int, *argtypes: Any, restype: Any = None) -> Any:
        assert index == win_uia._EL_CLASS_NAME

        def call(out: Any) -> int:
            bstr = api.SysAllocString("tiptap ProseMirror")
            allocated.append(bstr)
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = bstr
            return 0

        return call

    monkeypatch.setattr(win_uia, "_method", method)
    text = win_uia._read_bstr(Api(), 0x1, win_uia._EL_CLASS_NAME, "get_CurrentClassName")  # type: ignore[arg-type]
    assert text == "tiptap ProseMirror"
    assert freed == allocated
