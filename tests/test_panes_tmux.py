"""Tests for the tmux pane provider (panes/providers/tmux.py) with recorded CLI output."""

from __future__ import annotations

import pytest

from eye_tracker.panes.providers import tmux
from eye_tracker.panes.providers.command import CommandResult
from eye_tracker.panes.providers.tmux import (
    CLIENT_FORMAT,
    PANE_FORMAT,
    TmuxProvider,
    layout_panes,
    parse_clients,
    parse_panes,
)
from eye_tracker.panes.types import Pane, PaneError
from eye_tracker.types import AppIdentity, Rect, WindowRef
from panes_fakes import FakeRunner

TERMINAL_PID = 4000
CLIENT_PID = 4100
WINDOW = WindowRef(handle=0x3012A, pid=TERMINAL_PID, rect=Rect(92, 42, 1616, 916))
AREA = Rect(100, 50, 1600, 900)
APP = AppIdentity("windowsterminal", "CASCADIA_HOSTING_WINDOW_CLASS")

# `tmux list-clients -F CLIENT_FORMAT`: two terminals attached to one server.
CLIENTS = f"{CLIENT_PID}\t1727791200\t@1\t200\t50\tbottom\n5100\t1727791100\t@4\t120\t40\tbottom\n"
# `tmux list-panes -t @1 -F PANE_FORMAT`: one pane left, two stacked on the right,
# a 49-row window under a one-row status line.
PANES = "%0 0 0 100 49 1 200 49 0\n%1 101 0 99 24 0 200 49 0\n%2 101 25 99 24 0 200 49 0\n"


def provider(
    runner: FakeRunner,
    *,
    which: dict[str, str] | None = None,
    parents: dict[int, list[int]] | None = None,
    descendants: dict[int, set[str]] | None = None,
    system: str = "linux",
) -> TmuxProvider:
    tools = {"tmux": "/usr/bin/tmux"} if which is None else which
    tree = {CLIENT_PID: [3900, TERMINAL_PID, 1], 5100: [5000, 1]} if parents is None else parents
    below = descendants or {}
    return TmuxProvider(
        client_rect=lambda ref: AREA if ref.handle == WINDOW.handle else None,
        runner=runner,
        which=tools.get,
        parents=lambda pid: tree.get(pid, []),
        descendants=lambda pid: below.get(pid, set()),
        system=system,
        clock=lambda: 0.0,
    )


def native_runner() -> FakeRunner:
    return FakeRunner({("list-clients",): CLIENTS, ("list-panes",): PANES})


# ------------------------------------------------------------------ parsing
def test_parse_clients_skips_malformed_lines() -> None:
    clients = parse_clients(CLIENTS + "garbage\n12\tx\t@1\t1\t1\tbottom\n")
    assert [(c.pid, c.window_id, c.width, c.height) for c in clients] == [
        (CLIENT_PID, "@1", 200, 50),
        (5100, "@4", 120, 40),
    ]
    assert clients[0].status_top is False


def test_parse_panes_and_unreadable_output() -> None:
    panes = parse_panes(PANES)
    assert [(p.pane_id, p.left, p.top, p.width, p.height, p.active) for p in panes] == [
        ("%0", 0, 0, 100, 49, True),
        ("%1", 101, 0, 99, 24, False),
        ("%2", 101, 25, 99, 24, False),
    ]
    assert parse_panes("") == []
    with pytest.raises(PaneError):
        parse_panes("%0 0 0 100\n")  # an older tmux without some of the formats


def test_layout_maps_cells_onto_the_content_area() -> None:
    [client] = parse_clients(CLIENTS.splitlines()[0])
    panes = layout_panes(parse_panes(PANES), client, AREA)
    # 200 columns over 1600 px: 8 px per column; 50 rows over 900 px: 18 px per row.
    assert panes[0] == Pane("%0", Rect(100, 50, 800, 882), True, "tmux")
    assert panes[1] == Pane("%1", Rect(908, 50, 792, 432), False, "tmux")
    assert panes[2] == Pane("%2", Rect(908, 500, 792, 432), False, "tmux")


def test_layout_with_the_status_line_on_top() -> None:
    client = parse_clients("1\t0\t@1\t200\t50\ttop\n")[0]
    panes = layout_panes(parse_panes(PANES), client, AREA)
    assert panes[0].rect.y == 50 + 18  # below the status line


# ------------------------------------------------------------------ provider
def test_applies_to_terminals_with_tmux_only() -> None:
    p = provider(FakeRunner())
    assert p.applies(APP)
    assert p.applies(AppIdentity("kitty", "kitty"))
    assert not p.applies(AppIdentity("notepad", "Notepad"))
    assert not provider(FakeRunner(), which={}).applies(APP)  # no tmux anywhere


def test_detects_the_client_running_in_the_window() -> None:
    runner = native_runner()
    snapshot = provider(runner).detect(WINDOW, APP)
    assert snapshot is not None
    assert snapshot.window_handle == WINDOW.handle
    assert [p.id for p in snapshot.panes] == ["%0", "%1", "%2"]
    assert snapshot.focused is not None
    assert snapshot.focused.id == "%0"
    assert runner.commands() == [
        ("list-clients", "-F", CLIENT_FORMAT),
        ("list-panes", "-t", "@1", "-F", PANE_FORMAT),  # the window of *this* client
    ]
    assert all(call[0] == "/usr/bin/tmux" for call in runner.calls)


def test_no_snapshot_without_a_matching_client_or_server() -> None:
    other = WindowRef(handle=WINDOW.handle, pid=999)
    assert provider(native_runner()).detect(other, APP) is None
    no_server = FakeRunner({("list-clients",): CommandResult(1, "", "no server running")})
    assert provider(no_server).detect(WINDOW, APP) is None
    unknown_area = WindowRef(handle=1, pid=TERMINAL_PID)
    assert provider(native_runner()).detect(unknown_area, APP) is None


def test_single_or_zoomed_pane_is_no_snapshot() -> None:
    single = FakeRunner({("list-clients",): CLIENTS, ("list-panes",): "%0 0 0 200 49 1 200 49 0\n"})
    assert provider(single).detect(WINDOW, APP) is None
    zoomed = FakeRunner(
        {("list-clients",): CLIENTS, ("list-panes",): PANES.replace(" 0\n", " 1\n")}
    )
    assert provider(zoomed).detect(WINDOW, APP) is None


def test_focus_selects_the_pane_on_the_same_server() -> None:
    runner = native_runner()
    p = provider(runner)
    pane = Pane("%2", Rect(0, 0, 1, 1), False, "tmux")
    assert p.focus(WINDOW, pane) is False  # nothing detected for this window yet
    assert p.detect(WINDOW, APP) is not None
    runner.answers[("select-pane",)] = ""
    assert p.focus(WINDOW, pane) is True
    assert runner.calls[-1] == ["/usr/bin/tmux", "select-pane", "-t", "%2"]
    # Never anything but a pane id after -t.
    assert p.focus(WINDOW, Pane("; kill-server", Rect(0, 0, 1, 1), False, "tmux")) is False
    assert runner.calls[-1] == ["/usr/bin/tmux", "select-pane", "-t", "%2"]


def test_wsl_server_is_used_only_for_windows_running_wsl() -> None:
    runner = FakeRunner(
        {
            ("-e", "tmux", "list-clients"): CLIENTS.splitlines()[0],
            ("-e", "tmux", "list-panes"): PANES,
        }
    )
    wsl = "C:\\Windows\\System32\\wsl.exe"
    p = provider(
        runner,
        which={"wsl": wsl},
        parents={},
        descendants={TERMINAL_PID: {"openconsole", "wsl", "bash"}},
        system="win32",
    )
    assert p.applies(APP)
    snapshot = p.detect(WINDOW, APP)
    assert snapshot is not None
    assert runner.calls[0] == [wsl, "-e", "tmux", "list-clients", "-F", CLIENT_FORMAT]
    runner.answers[("-e", "tmux", "select-pane")] = ""
    assert p.focus(WINDOW, snapshot.panes[1])
    assert runner.calls[-1] == [wsl, "-e", "tmux", "select-pane", "-t", "%1"]

    # A window without WSL in its process tree never reaches the WSL server.
    plain = provider(FakeRunner(), which={"wsl": wsl}, parents={}, system="win32")
    assert plain.detect(WINDOW, APP) is None
    # WSL is a Windows thing.
    assert not provider(FakeRunner(), which={"wsl": wsl}, system="linux").applies(APP)


def test_two_wsl_clients_are_ambiguous() -> None:
    runner = FakeRunner({("-e", "tmux", "list-clients"): CLIENTS})
    p = provider(
        runner,
        which={"wsl": "wsl.exe"},
        parents={},
        descendants={TERMINAL_PID: {"wslhost"}},
        system="win32",
    )
    assert p.detect(WINDOW, APP) is None
    assert all("list-panes" not in call for call in runner.calls)


def test_runner_errors_propagate_as_pane_errors() -> None:
    runner = FakeRunner({("list-clients",): PaneError("tmux did not answer within 1 s")})
    with pytest.raises(PaneError):
        provider(runner).detect(WINDOW, APP)


def test_terminal_list_covers_the_common_emulators() -> None:
    assert {"windowsterminal", "wezterm-gui", "kitty", "iterm2", "ghostty"} <= (
        tmux.TERMINAL_PROCESSES
    )
