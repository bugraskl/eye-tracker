"""Running a terminal's own command-line tool, and looking at process trees.

Providers never run a shell: the argument vector goes straight to the program,
with a timeout, no console window (Windows) and no standard input. Everything
here is injectable so that tests never start a process.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ...platform.base import process_basename
from ...types import WindowRef
from ..types import PaneError

log = logging.getLogger(__name__)

#: Default time a provider command may take.
COMMAND_TIMEOUT_S = 1.0


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str], timeout: float) -> CommandResult: ...


def run_command(argv: Sequence[str], timeout: float = COMMAND_TIMEOUT_S) -> CommandResult:
    """Run ``argv`` (no shell) and return its exit code and output.

    Raises :class:`PaneError` when the program cannot be started or does not
    finish within ``timeout`` seconds (it is killed then).
    """
    args = [str(a) for a in argv]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        done = subprocess.run(
            args,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
    except subprocess.TimeoutExpired as exc:
        raise PaneError(f"{args[0]} did not answer within {timeout:g} s") from exc
    except OSError as exc:
        raise PaneError(f"{args[0]} could not be started: {exc}") from exc
    return CommandResult(done.returncode, done.stdout or "", done.stderr or "")


def window_key(ref: WindowRef) -> object:
    """A dictionary key for the window of ``ref``: its handle (when hashable) and
    its process id, so a handle reused by another process's window is another key."""
    try:
        hash(ref.handle)
    except TypeError:
        return (id(ref.handle), ref.pid)
    return (ref.handle, ref.pid)


def executable_of(pid: int) -> str | None:
    """Full path of the program ``pid`` runs (``None`` when gone or unreadable)."""
    try:
        import psutil

        exe = psutil.Process(pid).exe()
    except Exception:
        return None
    return str(exe) if exe else None


def parent_pids(pid: int) -> list[int]:
    """The ancestors of ``pid``, nearest first (``[]`` when it is gone or unreadable)."""
    try:
        import psutil

        return [int(p.pid) for p in psutil.Process(pid).parents()]
    except Exception:
        return []


def descendant_names(pid: int) -> set[str]:
    """Process names (see ``process_basename``) of everything ``pid`` started."""
    try:
        import psutil

        children = psutil.Process(pid).children(recursive=True)
    except Exception:
        return set()
    names: set[str] = set()
    for child in children:
        try:
            names.add(process_basename(str(child.name() or "")))
        except Exception:
            continue
    return names
