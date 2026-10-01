"""Tests for the WezTerm pane provider (panes/providers/wezterm.py) with recorded CLI output."""

from __future__ import annotations

import json
from typing import Any

import pytest

from eye_tracker.panes.providers.command import CommandResult
from eye_tracker.panes.providers.wezterm import (
    WezTermProvider,
    layout_panes,
    parse_clients,
    parse_panes,
)
from eye_tracker.panes.types import Pane, PaneError
from eye_tracker.types import AppIdentity, Rect, WindowRef
from panes_fakes import FakeRunner

GUI_PID = 4000
WINDOW = WindowRef(handle=0x5A0C, pid=GUI_PID, rect=Rect(92, 18, 1616, 940))
AREA = Rect(100, 50, 1600, 900)
APP = AppIdentity("wezterm-gui", "org.wezfurlong.wezterm")
CLI = "C:\\Program Files\\WezTerm\\wezterm.exe"
LIST = ("cli", "--no-auto-start", "list", "--format", "json")
CLIENTS_CMD = ("cli", "--no-auto-start", "list-clients", "--format", "json")


def pane_json(
    pane_id: int,
    *,
    tab_id: int = 0,
    window_id: int = 0,
    cols: int,
    rows: int = 52,
    left_col: int = 0,
    top_row: int = 0,
    is_active: bool = False,
    is_zoomed: bool = False,
) -> dict[str, Any]:
    """One entry of `wezterm cli list --format json` (8x16 px cells)."""
    return {
        "window_id": window_id,
        "tab_id": tab_id,
        "pane_id": pane_id,
        "workspace": "default",
        "size": {
            "rows": rows,
            "cols": cols,
            "pixel_width": cols * 8,
            "pixel_height": rows * 16,
            "dpi": 96,
        },
        "title": "vim notes.md",
        "cwd": "file:///C:/Users/me/secret-project/",
        "cursor_x": 3,
        "cursor_y": 10,
        "cursor_shape": "Default",
        "cursor_visibility": "Visible",
        "left_col": left_col,
        "top_row": top_row,
        "tab_title": "",
        "window_title": "vim notes.md",
        "is_active": is_active,
        "is_zoomed": is_zoomed,
        "tty_name": None,
    }


# Tab 0: two panes side by side (a one-column divider); tab 1: a single pane.
LISTED = json.dumps(
    [
        pane_json(0, cols=99, is_active=True),
        pane_json(1, cols=100, left_col=100),
        pane_json(2, tab_id=1, cols=200, is_active=True),
    ]
)
CLIENTS = json.dumps(
    [
        {
            "username": "me",
            "hostname": "desk",
            "pid": GUI_PID,
            "connection_elapsed": {"secs": 812, "nanos": 0},
            "idle_time": {"secs": 0, "nanos": 250000000},
            "workspace": "default",
            "focused_pane_id": 1,
        }
    ]
)


def provider(runner: FakeRunner, *, on_path: bool = True) -> WezTermProvider:
    def which(name: str) -> str | None:
        if name == "wezterm":
            return CLI if on_path else None
        return name if name.endswith("wezterm.exe") else None

    return WezTermProvider(
        client_rect=lambda ref: AREA if ref.handle == WINDOW.handle else None,
        runner=runner,
        which=which,
        executable=lambda pid: "C:\\Apps\\WezTerm\\wezterm-gui.exe" if pid == GUI_PID else None,
        system="win32",
    )


def runner(listed: str = LISTED, clients: str = CLIENTS) -> FakeRunner:
    return FakeRunner({LIST: listed, CLIENTS_CMD: clients})


# ------------------------------------------------------------------ parsing
def test_parse_panes_reads_only_geometry() -> None:
    panes = parse_panes(LISTED)
    assert [(p.pane_id, p.tab_id, p.cols, p.left_col, p.active) for p in panes] == [
        (0, 0, 99, 0, True),
        (1, 0, 100, 100, False),
        (2, 1, 200, 0, True),
    ]
    assert not hasattr(panes[0], "title")
    assert not hasattr(panes[0], "cwd")
    with pytest.raises(PaneError):
        parse_panes("not json")
    with pytest.raises(PaneError):
        parse_panes(json.dumps([{"pane_id": 1}]))
    assert parse_panes("[]") == []


def test_parse_clients() -> None:
    [client] = parse_clients(CLIENTS)
    assert (client.pid, client.focused_pane_id) == (GUI_PID, 1)
    assert client.idle_s == pytest.approx(0.25)
    assert parse_clients('[{"pid": "x"}, 5]') == []


def test_layout_uses_the_real_cell_size() -> None:
    tab = [p for p in parse_panes(LISTED) if p.tab_id == 0]
    left, right = layout_panes(tab, AREA)
    # The grid is 1600 x 832 px; the 68 px above it are the tab bar.
    assert left == Pane(0, Rect(100, 118, 792, 832), True, "wezterm")
    assert right == Pane(1, Rect(900, 118, 800, 832), False, "wezterm")


def test_layout_on_a_retina_display() -> None:
    tab = [p for p in parse_panes(LISTED) if p.tab_id == 0]
    left, right = layout_panes(tab, Rect(0, 25, 800, 450))  # points, half the pixels
    assert left.rect == Rect(0, 59, 396, 416)
    assert right.rect == Rect(400, 59, 400, 416)


# ------------------------------------------------------------------ provider
def test_applies_to_the_wezterm_gui_only() -> None:
    p = provider(runner())
    assert p.applies(APP)
    assert not p.applies(AppIdentity("windowsterminal", ""))


def test_detects_the_tab_with_the_guis_focused_pane() -> None:
    r = runner()
    snapshot = provider(r).detect(WINDOW, APP)
    assert snapshot is not None
    assert [p.id for p in snapshot.panes] == [0, 1]
    # The GUI focuses pane 1, whatever is_active says.
    assert snapshot.focused is not None
    assert snapshot.focused.id == 1
    assert r.commands() == [LIST, CLIENTS_CMD]
    assert r.calls[0][0] == CLI


def test_without_focus_information_only_an_unambiguous_tab_is_used() -> None:
    assert provider(runner(clients="[]")).detect(WINDOW, APP) is None  # two tabs
    one_tab = json.dumps(
        [pane_json(0, cols=99, is_active=True), pane_json(1, cols=100, left_col=100)]
    )
    snapshot = provider(runner(listed=one_tab, clients="[]")).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.focused is not None
    assert snapshot.focused.id == 0


def test_single_pane_zoom_or_no_gui_is_no_snapshot() -> None:
    clients = json.dumps([{"pid": GUI_PID, "focused_pane_id": 2}])
    assert provider(runner(clients=clients)).detect(WINDOW, APP) is None  # tab 1: one pane
    zoomed = json.dumps(
        [
            pane_json(0, cols=99, is_zoomed=True, is_active=True),
            pane_json(1, cols=100, left_col=100),
        ]
    )
    assert provider(runner(listed=zoomed, clients="[]")).detect(WINDOW, APP) is None
    down = FakeRunner({LIST: CommandResult(1, "", "failed to connect"), CLIENTS_CMD: "[]"})
    assert provider(down).detect(WINDOW, APP) is None
    assert down.commands() == [LIST]  # nothing else once the GUI does not answer


def test_cli_found_beside_the_gui_when_not_on_path() -> None:
    r = runner()
    assert provider(r, on_path=False).detect(WINDOW, APP) is not None
    assert r.calls[0][0] == "C:\\Apps\\WezTerm\\wezterm.exe"


def test_focus_activates_the_pane() -> None:
    r = runner()
    p = provider(r)
    target = Pane(0, Rect(0, 0, 1, 1), False, "wezterm")
    assert p.focus(WINDOW, target) is False  # not detected yet: no CLI known
    assert p.detect(WINDOW, APP) is not None
    r.answers[("cli", "--no-auto-start", "activate-pane")] = ""
    assert p.focus(WINDOW, target) is True
    assert r.calls[-1] == [CLI, "cli", "--no-auto-start", "activate-pane", "--pane-id", "0"]
    assert p.focus(WINDOW, Pane("0; rm", Rect(0, 0, 1, 1), False, "wezterm")) is False
