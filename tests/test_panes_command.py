"""Tests for the provider command runner and process helpers (panes/providers/command.py).

``subprocess.run`` and psutil are replaced: no process is started.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from eye_tracker.panes.providers import command
from eye_tracker.panes.providers.command import (
    CommandResult,
    descendant_names,
    parent_pids,
    run_command,
    window_key,
)
from eye_tracker.panes.types import PaneError
from eye_tracker.types import WindowRef


def test_run_command_passes_an_argument_vector_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(args: list[str], **kwargs: Any) -> Any:
        seen["args"] = args
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="%0 0 0\n", stderr="")

    monkeypatch.setattr(command.subprocess, "run", fake_run)
    result = run_command(["tmux", "list-panes", "-F", "#{pane_id}"], timeout=0.7)
    assert result == CommandResult(0, "%0 0 0\n", "")
    assert result.ok
    assert seen["args"] == ["tmux", "list-panes", "-F", "#{pane_id}"]
    assert seen.get("shell", False) is False
    assert seen["timeout"] == 0.7
    assert seen["stdin"] is subprocess.DEVNULL
    if command.sys.platform == "win32":
        assert seen["creationflags"] & subprocess.CREATE_NO_WINDOW


def test_run_command_failures_are_pane_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(args: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(command.subprocess, "run", timeout)
    with pytest.raises(PaneError, match="did not answer"):
        run_command(["wezterm", "cli", "list"], timeout=1.0)

    def missing(args: list[str], **kwargs: Any) -> Any:
        raise FileNotFoundError(2, "not found")

    monkeypatch.setattr(command.subprocess, "run", missing)
    with pytest.raises(PaneError, match="could not be started"):
        run_command(["wezterm"], timeout=1.0)

    def failed(args: list[str], **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args, 1, stdout=None, stderr="no server running")

    monkeypatch.setattr(command.subprocess, "run", failed)
    assert run_command(["tmux"], timeout=1.0) == CommandResult(1, "", "no server running")


class FakeProcess:
    def __init__(self, pid: int) -> None:
        if pid == 404:
            raise ProcessLookupError(pid)
        self.pid = pid

    def parents(self) -> list[Any]:
        return [FakeProcess(self.pid - 1), FakeProcess(1)]

    def children(self, recursive: bool = False) -> list[Any]:
        assert recursive
        return [_Named("OpenConsole.exe"), _Named("wsl.exe"), _Named("bash")]


class _Named:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


def test_process_tree_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    monkeypatch.setattr(psutil, "Process", FakeProcess)
    assert parent_pids(50) == [49, 1]
    assert parent_pids(404) == []
    assert descendant_names(50) == {"openconsole", "wsl", "bash"}
    assert descendant_names(404) == set()


def test_window_key() -> None:
    assert window_key(WindowRef(handle=12)) == 12
    unhashable = WindowRef(handle=[1, 2])
    assert window_key(unhashable) == id(unhashable.handle)
