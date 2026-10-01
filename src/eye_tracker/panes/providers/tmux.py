"""tmux panes, through the ``tmux`` command (also inside WSL on Windows).

Detection asks the tmux server which clients are attached (``list-clients``)
and picks the one running in the focused terminal window: a client whose
process descends from the window's process. Inside WSL the Linux process ids
cannot be matched to Windows ones, so a WSL server is only used when the
window's process tree contains WSL and exactly one client is attached. The
panes of that client's current window (``list-panes``) are then laid out over
the terminal's content area in proportion to their cell positions. Focus moves
with ``select-pane``. Only the default server socket is used; titles, commands
and working directories are never asked for.

The cell grid is assumed to fill the content area; a terminal that draws its
own tab bar or header bar there shifts the panes by that bar's height, which
the precision gate in the decider absorbs for panes of a useful size.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...types import AppIdentity, Rect, WindowRef
from ..types import Pane, PaneError, PaneSnapshot
from .command import (
    COMMAND_TIMEOUT_S,
    CommandRunner,
    descendant_names,
    parent_pids,
    run_command,
    window_key,
)

log = logging.getLogger(__name__)

#: Terminal emulators whose windows may run a tmux client (process names).
TERMINAL_PROCESSES: frozenset[str] = frozenset(
    {
        "windowsterminal",
        "wezterm-gui",
        "alacritty",
        "kitty",
        "gnome-terminal-server",
        "konsole",
        "xterm",
        "iterm2",
        "terminal",
        "foot",
        "ghostty",
        "tilix",
        "xfce4-terminal",
    }
)

#: Processes that show a terminal window runs WSL.
WSL_PROCESSES: frozenset[str] = frozenset({"wsl", "wslhost"})

CLIENT_FORMAT = (
    "#{client_pid}\t#{client_activity}\t#{window_id}\t#{client_width}\t#{client_height}"
    "\t#{status-position}"
)
PANE_FORMAT = (
    "#{pane_id} #{pane_left} #{pane_top} #{pane_width} #{pane_height} #{pane_active}"
    " #{window_width} #{window_height} #{window_zoomed_flag}"
)

#: How long a window's "runs WSL" answer is reused (walking a process tree is slow).
WSL_CHECK_TTL_S = 10.0

_PANE_ID = re.compile(r"^%\d+$")
_WINDOW_ID = re.compile(r"^@\d+$")


@dataclass(frozen=True, slots=True)
class TmuxClient:
    pid: int
    activity: int
    window_id: str
    width: int
    height: int
    status_top: bool


@dataclass(frozen=True, slots=True)
class TmuxPane:
    pane_id: str
    left: int
    top: int
    width: int
    height: int
    active: bool
    window_width: int
    window_height: int
    zoomed: bool


def parse_clients(text: str) -> list[TmuxClient]:
    """``list-clients -F CLIENT_FORMAT`` output; malformed lines are skipped."""
    clients = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 6 or not _WINDOW_ID.match(parts[2]):
            continue
        try:
            clients.append(
                TmuxClient(
                    pid=int(parts[0]),
                    activity=int(parts[1] or 0),
                    window_id=parts[2],
                    width=int(parts[3]),
                    height=int(parts[4]),
                    status_top=parts[5].strip() == "top",
                )
            )
        except ValueError:
            continue
    return clients


def parse_panes(text: str) -> list[TmuxPane]:
    """``list-panes -F PANE_FORMAT`` output. Raises :class:`PaneError` when the
    output is not empty but no line can be read (another tmux format)."""
    panes = []
    lines = [line for line in text.splitlines() if line.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) != 9 or not _PANE_ID.match(parts[0]):
            continue
        try:
            left, top, width, height, active, ww, wh, zoomed = (int(v) for v in parts[1:])
        except ValueError:
            continue
        if width <= 0 or height <= 0 or ww <= 0 or wh <= 0:
            continue
        panes.append(TmuxPane(parts[0], left, top, width, height, active == 1, ww, wh, zoomed == 1))
    if lines and not panes:
        raise PaneError("tmux list-panes output could not be read")
    return panes


def layout_panes(
    panes: Sequence[TmuxPane], client: TmuxClient, area: Rect, provider: str = "tmux"
) -> tuple[Pane, ...]:
    """Map pane cells onto the terminal's content area ``area``."""
    first = panes[0]
    cols = max(client.width, first.window_width)
    rows = max(client.height, first.window_height)
    top_rows = rows - first.window_height if client.status_top else 0
    cw = area.w / cols
    ch = area.h / rows

    def span(start: int, size: int, origin: int, cell: float) -> tuple[int, int]:
        a = origin + round(start * cell)
        b = origin + round((start + size) * cell)
        return a, max(1, b - a)

    out = []
    for p in panes:
        x, w = span(p.left, p.width, area.x, cw)
        y, h = span(top_rows + p.top, p.height, area.y, ch)
        out.append(Pane(p.pane_id, Rect(x, y, w, h), p.active, provider))
    return tuple(out)


class TmuxProvider:
    """Panes of a tmux client running in the focused terminal window."""

    name = "tmux"

    def __init__(
        self,
        *,
        client_rect: Callable[[WindowRef], Rect | None],
        runner: CommandRunner = run_command,
        which: Callable[[str], str | None] = shutil.which,
        parents: Callable[[int], list[int]] = parent_pids,
        descendants: Callable[[int], set[str]] = descendant_names,
        system: str = sys.platform,
        clock: Callable[[], float] = time.monotonic,
        timeout: float = COMMAND_TIMEOUT_S,
    ) -> None:
        self._client_rect = client_rect
        self._run = runner
        self._which = which
        self._parents = parents
        self._descendants = descendants
        self._system = system
        self._clock = clock
        self._timeout = timeout
        self._tools: dict[str, str | None] = {}
        self._wsl_windows: dict[int, tuple[float, bool]] = {}
        #: Command prefix that reached the server of each window's panes.
        self._prefixes: dict[object, list[str]] = {}

    # ------------------------------------------------------------- protocol
    def applies(self, app: AppIdentity) -> bool:
        if app.process not in TERMINAL_PROCESSES:
            return False
        return self._tool("tmux") is not None or self._wsl_tool() is not None

    def detect(self, ref: WindowRef, app: AppIdentity) -> PaneSnapshot | None:
        area = self._client_rect(ref)
        if area is None or area.w <= 0 or area.h <= 0:
            return None
        found = self._client_for(ref)
        if found is None:
            self._prefixes.pop(window_key(ref), None)
            return None
        prefix, client = found
        result = self._run(
            [*prefix, "list-panes", "-t", client.window_id, "-F", PANE_FORMAT], self._timeout
        )
        if not result.ok:
            return None  # the window closed between the two commands
        panes = parse_panes(result.stdout)
        if len(panes) < 2 or any(p.zoomed for p in panes):
            return None  # nothing to choose between (a zoomed pane fills the window)
        if len(self._prefixes) > 64:
            self._prefixes.clear()
        self._prefixes[window_key(ref)] = prefix
        return PaneSnapshot(ref.handle, layout_panes(panes, client, area, self.name), 0.0)

    def focus(self, ref: WindowRef, pane: Pane) -> bool:
        pane_id = str(pane.id)
        prefix = self._prefixes.get(window_key(ref))
        if prefix is None or not _PANE_ID.match(pane_id):
            return False
        return self._run([*prefix, "select-pane", "-t", pane_id], self._timeout).ok

    # ------------------------------------------------------------- internals
    def _tool(self, name: str) -> str | None:
        if name not in self._tools:
            self._tools[name] = self._which(name)
        return self._tools[name]

    def _wsl_tool(self) -> str | None:
        if not self._system.startswith("win"):
            return None
        return self._tool("wsl")

    def _client_for(self, ref: WindowRef) -> tuple[list[str], TmuxClient] | None:
        tmux = self._tool("tmux")
        if tmux is not None and ref.pid is not None:
            clients = self._list_clients([tmux])
            ancestors_of = {c.pid: set(self._parents(c.pid)) for c in clients}
            mine = [c for c in clients if ref.pid in ancestors_of[c.pid]]
            if mine:
                return [tmux], max(mine, key=lambda c: c.activity)
        wsl = self._wsl_tool()
        if wsl is not None and ref.pid is not None and self._runs_wsl(ref.pid):
            prefix = [wsl, "-e", "tmux"]
            clients = self._list_clients(prefix)
            if len(clients) == 1:
                return prefix, clients[0]
            if clients:
                log.debug("%d tmux clients in WSL; cannot tell which is this window", len(clients))
        return None

    def _list_clients(self, prefix: list[str]) -> list[TmuxClient]:
        result = self._run([*prefix, "list-clients", "-F", CLIENT_FORMAT], self._timeout)
        if not result.ok:
            return []  # no server running
        return parse_clients(result.stdout)

    def _runs_wsl(self, pid: int) -> bool:
        now = self._clock()
        cached = self._wsl_windows.get(pid)
        if cached is not None and now - cached[0] < WSL_CHECK_TTL_S:
            return cached[1]
        runs = bool(self._descendants(pid) & WSL_PROCESSES)
        if len(self._wsl_windows) > 32:
            self._wsl_windows.clear()
        self._wsl_windows[pid] = (now, runs)
        return runs
