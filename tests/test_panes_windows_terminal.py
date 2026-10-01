"""Tests for the Windows Terminal pane provider (panes/providers/windows_terminal.py)
and its UI Automation binding (platform/win_uia.py).

The provider is tested everywhere with a fake UI Automation client; no test
queries a real window. The binding gets a Windows-only smoke test that creates
the automation object and nothing else.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from typing import Any

import pytest

from eye_tracker.panes.providers.windows_terminal import (
    PANE_CLASS,
    WINDOW_CLASS,
    WindowsTerminalProvider,
    pane_id,
)
from eye_tracker.panes.registry import PaneRegistry
from eye_tracker.panes.types import Pane, PaneError
from eye_tracker.panes.worker import FAILURE_LIMIT, PaneWorker
from eye_tracker.platform import win_uia
from eye_tracker.platform.win_uia import ComError, UiaElement, UiAutomation
from eye_tracker.types import AppIdentity, Rect, WindowRef

HWND = 0x40A2C
WINDOW = WindowRef(handle=HWND, pid=9000, rect=Rect(-8, -8, 1936, 1056))
AREA = Rect(0, 0, 1920, 1040)
APP = AppIdentity("windowsterminal", WINDOW_CLASS)

LEFT = UiaElement((42, HWND, 4, 101), Rect(8, 80, 948, 952), True, False)
RIGHT = UiaElement((42, HWND, 4, 102), Rect(964, 80, 948, 952), False, False)


class FakeUia:
    """Answers like UI Automation would; records every call."""

    def __init__(
        self,
        elements: list[UiaElement] | None = None,
        *,
        foreground: int | None = HWND,
        fail: BaseException | None = None,
    ) -> None:
        self.elements = list(elements if elements is not None else [LEFT, RIGHT])
        self.foreground = foreground
        self.fail = fail
        self.calls: list[tuple[Any, ...]] = []
        self.focused: list[UiaElement] = []
        #: Foreground windows answered first, one per call (then ``foreground``).
        self.foreground_sequence: list[int | None] = []
        self.closed = 0

    def find(self, hwnd: int, class_name: str) -> list[UiaElement]:
        self.calls.append(("find", hwnd, class_name))
        if self.fail is not None:
            raise self.fail
        return list(self.elements)

    def focus(
        self,
        hwnd: int,
        class_name: str,
        match: Callable[[UiaElement], bool],
        *,
        guard: Callable[[], bool] | None = None,
    ) -> bool:
        self.calls.append(("focus", hwnd, class_name))
        if self.fail is not None:
            raise self.fail
        for element in self.elements:
            if match(element):
                if guard is not None and not guard():
                    return False
                self.focused.append(element)
                return True
        return False

    def foreground_window(self) -> int | None:
        self.calls.append(("foreground",))
        if self.foreground_sequence:
            return self.foreground_sequence.pop(0)
        return self.foreground

    def close(self) -> None:
        self.closed += 1


def provider(client: FakeUia) -> WindowsTerminalProvider:
    return WindowsTerminalProvider(
        client_rect=lambda ref: AREA if ref.handle == HWND else None, client=client
    )


# ------------------------------------------------------------------ applies
def test_creating_the_provider_touches_nothing() -> None:
    p = WindowsTerminalProvider(client_rect=lambda ref: None, runner=None)
    assert p.name == "windows_terminal"
    assert p.applies(APP)
    assert p._client is None  # UI Automation only on the worker thread, when asked


def test_applies_to_windows_terminal_windows_only() -> None:
    p = provider(FakeUia())
    assert p.applies(APP)
    assert p.applies(AppIdentity("WindowsTerminal", WINDOW_CLASS))
    assert not p.applies(AppIdentity("windowsterminal", "PseudoConsoleWindow"))
    assert not p.applies(AppIdentity("openconsole", WINDOW_CLASS))
    assert not p.applies(AppIdentity("wezterm-gui", "org.wezfurlong.wezterm"))
    assert not p.applies(AppIdentity("code", "Chrome_WidgetWin_1"))


# ------------------------------------------------------------------- detect
def test_detects_two_panes_with_the_focused_one() -> None:
    client = FakeUia()
    snapshot = provider(client).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.window_handle == HWND
    assert [p.rect for p in snapshot.panes] == [LEFT.rect, RIGHT.rect]
    assert [p.id for p in snapshot.panes] == [LEFT.runtime_id, RIGHT.runtime_id]
    assert {p.provider for p in snapshot.panes} == {"windows_terminal"}
    assert snapshot.focused is not None
    assert snapshot.focused.id == LEFT.runtime_id
    assert client.calls == [("foreground",), ("find", HWND, PANE_CLASS)]


def test_three_panes_are_ordered_top_to_bottom_then_left_to_right() -> None:
    top_right = UiaElement((42, 1, 3), Rect(964, 80, 948, 470), False, False)
    bottom_right = UiaElement((42, 1, 4), Rect(964, 556, 948, 476), True, False)
    left = UiaElement((42, 1, 2), Rect(8, 80, 948, 952), False, False)
    snapshot = provider(FakeUia([bottom_right, left, top_right])).detect(WINDOW, APP)
    assert snapshot is not None
    assert [p.id for p in snapshot.panes] == [(42, 1, 2), (42, 1, 3), (42, 1, 4)]
    assert snapshot.focused is not None
    assert snapshot.focused.id == (42, 1, 4)


def test_offscreen_and_empty_panes_are_ignored() -> None:
    hidden = UiaElement((42, 1, 9), Rect(8, 80, 948, 952), False, True)
    empty = UiaElement((42, 1, 8), None, False, False)
    outside = UiaElement((42, 1, 7), Rect(3000, 80, 500, 500), False, False)
    snapshot = provider(FakeUia([LEFT, hidden, empty, outside, RIGHT])).detect(WINDOW, APP)
    assert snapshot is not None
    assert [p.id for p in snapshot.panes] == [LEFT.runtime_id, RIGHT.runtime_id]
    # Only one visible pane left: no snapshot (tmux in it may then be asked).
    assert provider(FakeUia([LEFT, hidden, empty, outside])).detect(WINDOW, APP) is None
    assert provider(FakeUia([])).detect(WINDOW, APP) is None


def test_pane_rectangles_are_cut_to_the_window() -> None:
    wide = UiaElement((42, 1, 1), Rect(-20, 80, 990, 952), True, False)
    snapshot = provider(FakeUia([wide, RIGHT])).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.panes[0].rect == Rect(0, 80, 970, 952)


def test_no_focus_flag_when_no_pane_has_the_keyboard_focus() -> None:
    unfocused = UiaElement(LEFT.runtime_id, LEFT.rect, False, False)
    snapshot = provider(FakeUia([unfocused, RIGHT])).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.focused is None


def test_ids_are_stable_across_snapshots() -> None:
    client = FakeUia()
    p = provider(client)
    first = p.detect(WINDOW, APP)
    # The user focused the other pane and resized the split: same ids.
    client.elements = [
        UiaElement(LEFT.runtime_id, Rect(8, 80, 600, 952), False, False),
        UiaElement(RIGHT.runtime_id, Rect(616, 80, 1296, 952), True, False),
    ]
    second = p.detect(WINDOW, APP)
    assert first is not None
    assert second is not None
    assert [x.id for x in first.panes] == [x.id for x in second.panes]
    assert second.focused is not None
    assert second.focused.id == RIGHT.runtime_id


def test_without_a_runtime_id_the_rectangle_is_the_id() -> None:
    left = UiaElement(None, LEFT.rect, True, False)
    right = UiaElement((), RIGHT.rect, False, False)
    client = FakeUia([left, right])
    p = provider(client)
    snapshot = p.detect(WINDOW, APP)
    assert snapshot is not None
    assert [x.id for x in snapshot.panes] == [
        ("rect", 8, 80, 948, 952),
        ("rect", 964, 80, 948, 952),
    ]
    assert p.detect(WINDOW, APP) == snapshot  # stable while the layout is
    assert pane_id(UiaElement(None, None, False, False)) is None


def test_duplicate_ids_count_once() -> None:
    assert provider(FakeUia([LEFT, LEFT])).detect(WINDOW, APP) is None


def test_nothing_is_asked_unless_the_window_is_in_the_foreground() -> None:
    client = FakeUia(foreground=0x1234)
    p = provider(client)
    assert p.detect(WINDOW, APP) is None
    assert p.focus(WINDOW, Pane(RIGHT.runtime_id, RIGHT.rect, False, "windows_terminal")) is False
    assert client.calls == [("foreground",), ("foreground",)]
    assert client.focused == []
    assert p.detect(WindowRef(handle=None), APP) is None


# -------------------------------------------------------------------- focus
def test_focus_sets_focus_on_the_matching_pane() -> None:
    client = FakeUia()
    p = provider(client)
    snapshot = p.detect(WINDOW, APP)
    assert snapshot is not None
    target = snapshot.panes[1]
    assert p.focus(WINDOW, target) is True
    assert client.focused == [RIGHT]
    # The foreground is checked again inside the walk, right before SetFocus.
    assert client.calls[-2:] == [("focus", HWND, PANE_CLASS), ("foreground",)]


def test_focus_of_a_pane_that_is_gone_fails() -> None:
    client = FakeUia()
    p = provider(client)
    gone = Pane((42, HWND, 4, 999), RIGHT.rect, False, "windows_terminal")
    assert p.focus(WINDOW, gone) is False
    assert client.focused == []
    # An off-screen element is never focused, even with the right id.
    client.elements = [LEFT, UiaElement(RIGHT.runtime_id, RIGHT.rect, False, True)]
    assert p.focus(WINDOW, Pane(RIGHT.runtime_id, RIGHT.rect, False, "windows_terminal")) is False
    # Another provider's pane is not ours to focus.
    assert p.focus(WINDOW, Pane(RIGHT.runtime_id, RIGHT.rect, False, "tmux")) is False


def test_no_focus_when_another_window_came_to_the_front_during_the_walk() -> None:
    client = FakeUia()
    p = provider(client)
    target = Pane(RIGHT.runtime_id, RIGHT.rect, False, "windows_terminal")
    client.foreground_sequence = [HWND, 0x1234]  # in front before the walk, not after it
    assert p.focus(WINDOW, target) is False
    assert client.focused == []
    # The second check runs inside the walk, right before SetFocus.
    assert client.calls == [("foreground",), ("focus", HWND, PANE_CLASS), ("foreground",)]
    assert p.focus(WINDOW, target) is True  # still in front: focused


def test_close_releases_the_automation_client() -> None:
    client = FakeUia()
    p = provider(client)
    p.close()
    assert client.closed == 1
    p.close()  # nothing left to release
    assert client.closed == 1


def test_binding_focus_asks_the_guard_right_before_set_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UiAutomation.focus with the COM layer replaced: SetFocus only if the guard agrees."""
    calls: list[str] = []

    def fake_method(ptr: int, index: int, *argtypes: Any, restype: Any = None) -> Any:
        assert index == win_uia._EL_SET_FOCUS
        return lambda: calls.append(f"SetFocus {ptr}") or 0

    def fake_each(hwnd: int, class_name: str, visit: Callable[[int, UiaElement], bool]) -> None:
        for ptr, info in ((11, LEFT), (12, RIGHT)):
            calls.append(f"visit {ptr}")
            if visit(ptr, info):
                return

    monkeypatch.setattr(win_uia, "_method", fake_method)
    client = UiAutomation()
    monkeypatch.setattr(client, "_each", fake_each)

    def is_right(element: UiaElement) -> bool:
        return element is RIGHT

    def refuse() -> bool:
        calls.append("guard")
        return False

    assert client.focus(HWND, PANE_CLASS, is_right, guard=refuse) is False
    assert calls == ["visit 11", "visit 12", "guard"]  # no SetFocus
    calls.clear()
    assert client.focus(HWND, PANE_CLASS, is_right, guard=lambda: True) is True
    assert calls == ["visit 11", "visit 12", "SetFocus 12"]


def test_com_errors_become_pane_errors() -> None:
    client = FakeUia(fail=ComError("IUIAutomationElement::FindAll", 0x80131505))
    p = provider(client)
    with pytest.raises(PaneError, match="0x80131505"):
        p.detect(WINDOW, APP)
    with pytest.raises(PaneError):
        p.focus(WINDOW, Pane(RIGHT.runtime_id, RIGHT.rect, False, "windows_terminal"))


# -------------------------------------------------------- with the worker
def _worker(client: FakeUia) -> tuple[PaneWorker, list[Any], list[Any]]:
    worker = PaneWorker(PaneRegistry([provider(client)]), synchronous=True)
    detected: list[Any] = []
    focused: list[Any] = []
    worker.detected.connect(detected.append)
    worker.focused.connect(focused.append)
    return worker, detected, focused


def test_worker_switches_the_provider_off_after_three_com_errors(qapp: Any) -> None:
    client = FakeUia(fail=ComError("IUIAutomation::ElementFromHandle", 0x80040201))
    worker, detected, focused = _worker(client)
    for _ in range(FAILURE_LIMIT + 2):
        worker.request_detect(WINDOW, APP)
    assert sum(1 for c in client.calls if c[0] == "find") == FAILURE_LIMIT
    assert worker.disabled_providers == frozenset({"windows_terminal"})
    assert all(r.snapshot is None for r in detected)
    calls = len(client.calls)
    worker.request_focus(WINDOW, Pane(RIGHT.runtime_id, RIGHT.rect, False, "windows_terminal"))
    assert focused[-1].ok is False
    assert len(client.calls) == calls  # not asked any more


def test_worker_detects_and_focuses(qapp: Any) -> None:
    client = FakeUia()
    worker, detected, focused = _worker(client)
    worker.request_detect(WINDOW, APP)
    [result] = detected
    assert result.snapshot is not None
    assert result.snapshot.provider == "windows_terminal"
    worker.request_focus(WINDOW, result.snapshot.panes[1])
    assert focused[-1].ok is True
    assert client.focused == [RIGHT]


@pytest.mark.parametrize(
    "app",
    [
        AppIdentity("code", "Chrome_WidgetWin_1"),
        AppIdentity("cursor", "Chrome_WidgetWin_1"),
        AppIdentity("claude", "Chrome_WidgetWin_1"),
        AppIdentity("Code", "Chrome_WidgetWin_1"),
        AppIdentity("someapp", "Chrome_WidgetWin_1"),
        # Even a Windows Terminal name on a Chromium window class is not asked.
        AppIdentity("windowsterminal", "Chrome_WidgetWin_1"),
    ],
    ids=lambda a: f"{a.process}|{a.app_id}",
)
def test_denied_apps_never_reach_ui_automation(qapp: Any, app: AppIdentity) -> None:
    client = FakeUia()
    worker, detected, _ = _worker(client)
    worker.request_detect(WINDOW, app)
    assert client.calls == []
    assert detected[0].snapshot is None


# ------------------------------------------------------------------ binding
def test_binding_imports_everywhere_and_loads_nothing_until_used() -> None:
    client = UiAutomation()
    assert isinstance(client, UiAutomation)
    error = ComError("FindAll", -2146233083)  # 0x80131505 as a signed HRESULT
    assert isinstance(error, OSError)
    assert error.hresult == 0x80131505
    assert "0x80131505" in str(error)


def test_binding_refuses_other_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(OSError, match="only available on Windows"):
        UiAutomation().find(HWND, PANE_CLASS)
    with pytest.raises(OSError, match="only available on Windows"):
        UiAutomation().foreground_window()


def _on_new_thread(fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on a fresh thread (COM gets a multithreaded apartment there)."""
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(10.0)
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


@pytest.mark.skipif(sys.platform != "win32", reason="needs COM and UI Automation")
def test_smoke_creates_the_automation_object_without_asking_any_window() -> None:
    def create() -> tuple[int, int] | None:
        client = UiAutomation(connection_timeout_ms=210, transaction_timeout_ms=480)
        try:
            # Reading the timeouts back proves the IUIAutomation2 vtable slots.
            return client.timeouts()
        finally:
            client.close()

    assert _on_new_thread(create) == (210, 480)


@pytest.mark.skipif(sys.platform != "win32", reason="needs oleaut32")
def test_smoke_reads_an_integer_safearray() -> None:
    import ctypes

    api = win_uia._api()
    oleaut32 = ctypes.WinDLL("oleaut32")
    create = oleaut32.SafeArrayCreateVector
    create.restype = ctypes.c_void_p
    create.argtypes = [ctypes.c_ushort, ctypes.c_long, ctypes.c_ulong]
    psa = create(win_uia.VT_I4, 0, 3)
    data = ctypes.c_void_p()
    assert api.SafeArrayAccessData(psa, ctypes.byref(data)) == 0
    values = (ctypes.c_int * 3).from_address(data.value or 0)
    values[:] = [42, 0x40A2C, -3]
    api.SafeArrayUnaccessData(psa)
    assert win_uia.read_int_safearray(api, psa) == (42, 0x40A2C, -3)  # and destroyed
    other = create(8, 0, 1)  # VT_BSTR: not a runtime id
    assert win_uia.read_int_safearray(api, other) is None
    assert win_uia.read_int_safearray(api, 0) is None
