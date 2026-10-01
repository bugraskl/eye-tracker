"""WezTerm panes, through ``wezterm cli`` (never starting a mux server).

``wezterm cli --no-auto-start list --format json`` lists every pane with its
cell position and size, including pixel sizes; ``list-clients`` tells which
pane the GUI has focused, and so which window and tab the user sees. The panes
of that tab are laid out over the window's content area with the real cell
size in pixels (scaled down on Retina displays, where WezTerm counts device
pixels and the desktop counts points). Focus moves with ``activate-pane``.
Only the needed fields are read; titles and working directories are ignored.

The padding around the grid is assumed equal on the left, right and bottom, and
everything else above the grid is taken to be the tab bar (WezTerm's default
layout). A tab bar at the bottom shifts the panes by its height.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import ntpath
import posixpath
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ...types import AppIdentity, Rect, WindowRef
from ..types import Pane, PaneError, PaneSnapshot
from .command import (
    COMMAND_TIMEOUT_S,
    CommandRunner,
    executable_of,
    run_command,
    window_key,
)

log = logging.getLogger(__name__)

GUI_PROCESSES: frozenset[str] = frozenset({"wezterm-gui"})


@dataclass(frozen=True, slots=True)
class WezPane:
    window_id: int
    tab_id: int
    pane_id: int
    rows: int
    cols: int
    pixel_width: int
    pixel_height: int
    left_col: int
    top_row: int
    active: bool
    zoomed: bool


@dataclass(frozen=True, slots=True)
class WezClient:
    pid: int
    idle_s: float
    focused_pane_id: int | None


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"not an integer: {value!r}")
    return value


def parse_panes(text: str) -> list[WezPane]:
    """``wezterm cli list --format json`` output. Raises :class:`PaneError` if unreadable."""
    try:
        data = json.loads(text or "[]")
    except ValueError as exc:
        raise PaneError("wezterm cli list output is not JSON") from exc
    if not isinstance(data, list):
        raise PaneError("wezterm cli list output is not a list")
    panes = []
    for item in data:
        try:
            size = item["size"]
            pane = WezPane(
                window_id=_int(item["window_id"]),
                tab_id=_int(item["tab_id"]),
                pane_id=_int(item["pane_id"]),
                rows=_int(size["rows"]),
                cols=_int(size["cols"]),
                pixel_width=_int(size["pixel_width"]),
                pixel_height=_int(size["pixel_height"]),
                left_col=_int(item["left_col"]),
                top_row=_int(item["top_row"]),
                active=bool(item.get("is_active", False)),
                zoomed=bool(item.get("is_zoomed", False)),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if min(pane.rows, pane.cols, pane.pixel_width, pane.pixel_height) > 0:
            panes.append(pane)
    if data and not panes:
        raise PaneError("wezterm cli list output could not be read")
    return panes


def parse_clients(text: str) -> list[WezClient]:
    """``wezterm cli list-clients --format json`` output (unreadable entries skipped)."""
    try:
        data = json.loads(text or "[]")
    except ValueError as exc:
        raise PaneError("wezterm cli list-clients output is not JSON") from exc
    clients = []
    for item in data if isinstance(data, list) else []:
        try:
            idle = item.get("idle_time") or {}
            focused = item.get("focused_pane_id")
            clients.append(
                WezClient(
                    pid=_int(item["pid"]),
                    idle_s=float(idle.get("secs", 0)) + float(idle.get("nanos", 0)) / 1e9,
                    focused_pane_id=None if focused is None else _int(focused),
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return clients


def layout_panes(
    panes: Sequence[WezPane], area: Rect, provider: str = "wezterm"
) -> tuple[Pane, ...]:
    """Place the panes of one tab on the window's content area ``area``."""
    first = panes[0]
    cell_w = first.pixel_width / first.cols
    cell_h = first.pixel_height / first.rows
    grid_w = max(p.left_col + p.cols for p in panes) * cell_w
    grid_h = max(p.top_row + p.rows for p in panes) * cell_h
    # Device pixels against desktop points (macOS Retina): a whole-number ratio.
    ratio = max(1, round(grid_w / area.w)) if grid_w > area.w * 1.02 else 1
    scale = 1.0 / ratio
    pad_x = max(0.0, (area.w - grid_w * scale) / 2.0)
    pad_top = max(0.0, area.h - grid_h * scale - pad_x)
    out = []
    for p in panes:
        x0 = area.x + pad_x + p.left_col * cell_w * scale
        y0 = area.y + pad_top + p.top_row * cell_h * scale
        x1 = x0 + p.cols * cell_w * scale
        y1 = y0 + p.rows * cell_h * scale
        rect = Rect(
            round(x0), round(y0), max(1, round(x1) - round(x0)), max(1, round(y1) - round(y0))
        )
        out.append(Pane(p.pane_id, rect, p.active, provider))
    return tuple(out)


class WezTermProvider:
    """Panes of the tab a WezTerm GUI window shows."""

    name = "wezterm"

    def __init__(
        self,
        *,
        client_rect: Callable[[WindowRef], Rect | None],
        runner: CommandRunner = run_command,
        which: Callable[[str], str | None] = shutil.which,
        executable: Callable[[int], str | None] = executable_of,
        system: str = sys.platform,
        timeout: float = COMMAND_TIMEOUT_S,
    ) -> None:
        self._client_rect = client_rect
        self._run = runner
        self._which = which
        self._executable = executable
        self._system = system
        self._timeout = timeout
        self._path_checked = False
        self._on_path: str | None = None
        self._cli: dict[object, str] = {}

    def applies(self, app: AppIdentity) -> bool:
        return app.process in GUI_PROCESSES

    def detect(self, ref: WindowRef, app: AppIdentity) -> PaneSnapshot | None:
        area = self._client_rect(ref)
        if area is None or area.w <= 0 or area.h <= 0:
            return None
        cli = self._cli_for(ref)
        if cli is None:
            return None
        listed = self._run(
            [cli, "cli", "--no-auto-start", "list", "--format", "json"], self._timeout
        )
        if not listed.ok:
            return None  # no GUI or mux to talk to
        panes = parse_panes(listed.stdout)
        clients = self._run(
            [cli, "cli", "--no-auto-start", "list-clients", "--format", "json"], self._timeout
        )
        focused = self._focused_pane(parse_clients(clients.stdout) if clients.ok else [], ref)
        tab = self._visible_tab(panes, focused)
        if len(tab) < 2 or any(p.zoomed for p in tab):
            return None
        if focused is not None:
            # The GUI's focus is the truth; is_active is per tab.
            tab = [dataclasses.replace(p, active=p.pane_id == focused) for p in tab]
        return PaneSnapshot(ref.handle, layout_panes(tab, area, self.name), 0.0)

    def focus(self, ref: WindowRef, pane: Pane) -> bool:
        cli = self._cli.get(window_key(ref))
        if cli is None or isinstance(pane.id, bool) or not isinstance(pane.id, int):
            return False
        argv = [cli, "cli", "--no-auto-start", "activate-pane", "--pane-id", str(pane.id)]
        return self._run(argv, self._timeout).ok

    # ------------------------------------------------------------- internals
    def _cli_for(self, ref: WindowRef) -> str | None:
        """The ``wezterm`` program: on PATH, else beside the GUI's executable."""
        key = window_key(ref)
        if key in self._cli:
            return self._cli[key]
        if not self._path_checked:
            self._path_checked = True
            self._on_path = self._which("wezterm")
        cli = self._on_path
        if cli is None and ref.pid is not None:
            gui = self._executable(ref.pid)
            if gui:
                path = ntpath if self._system.startswith("win") else posixpath
                name = "wezterm.exe" if self._system.startswith("win") else "wezterm"
                candidate = path.join(path.dirname(gui), name)
                cli = self._which(candidate)
        if cli is None:
            return None
        if len(self._cli) > 64:
            self._cli.clear()
        self._cli[key] = cli
        return cli

    @staticmethod
    def _focused_pane(clients: Sequence[WezClient], ref: WindowRef) -> int | None:
        known = [c for c in clients if c.focused_pane_id is not None]
        own = [c for c in known if c.pid == ref.pid]
        pool = own or known
        if not pool:
            return None
        return min(pool, key=lambda c: c.idle_s).focused_pane_id

    @staticmethod
    def _visible_tab(panes: Sequence[WezPane], focused: int | None) -> list[WezPane]:
        if focused is not None:
            owner = next((p for p in panes if p.pane_id == focused), None)
            if owner is None:
                return []
            key = (owner.window_id, owner.tab_id)
        else:
            tabs = {(p.window_id, p.tab_id) for p in panes}
            if len(tabs) != 1:
                return []  # which tab is on screen cannot be told
            key = next(iter(tabs))
        return [p for p in panes if (p.window_id, p.tab_id) == key]
