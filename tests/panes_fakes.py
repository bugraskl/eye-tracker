"""Fakes shared by the split-pane tests (not a test module itself).

Nothing here starts a process: command lines are answered from a table.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from eye_tracker.panes.providers.command import CommandResult
from eye_tracker.panes.types import Pane, PaneError, PaneSnapshot
from eye_tracker.types import AppIdentity, Rect, WindowRef


class FakeRunner:
    """Answers command lines by their arguments after the program (``argv[1:]``).

    A key is matched as a prefix of the arguments, so ``("list-panes",)`` answers
    every ``list-panes`` call. A value may be a :class:`CommandResult`, a string
    (stdout of a successful run) or an exception to raise.
    """

    def __init__(self, answers: dict[tuple[str, ...], Any] | None = None) -> None:
        self.answers: dict[tuple[str, ...], Any] = dict(answers or {})
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> CommandResult:
        args = [str(a) for a in argv]
        self.calls.append(args)
        rest = tuple(args[1:])
        best: tuple[str, ...] | None = None
        for key in self.answers:
            if rest[: len(key)] == key and (best is None or len(key) > len(best)):
                best = key
        if best is None:
            return CommandResult(1, "", "unknown command")
        answer = self.answers[best]
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, CommandResult):
            return answer
        return CommandResult(0, str(answer), "")

    def commands(self) -> list[tuple[str, ...]]:
        return [tuple(c[1:]) for c in self.calls]


class RecordingProvider:
    """A provider that records every call (to prove what is never asked)."""

    def __init__(
        self,
        name: str = "fake",
        *,
        applies: bool = True,
        snapshot: Callable[[WindowRef], PaneSnapshot | None] | None = None,
        focus_ok: bool = True,
        fail: BaseException | None = None,
    ) -> None:
        self.name = name
        self._applies = applies
        self._snapshot = snapshot
        self._focus_ok = focus_ok
        self.fail = fail
        self.calls: list[tuple[str, Any]] = []
        self.closed_on: list[str] = []

    def applies(self, app: AppIdentity) -> bool:
        self.calls.append(("applies", app))
        return self._applies

    def detect(self, ref: WindowRef, app: AppIdentity) -> PaneSnapshot | None:
        self.calls.append(("detect", ref.handle))
        if self.fail is not None:
            raise self.fail
        return self._snapshot(ref) if self._snapshot is not None else None

    def focus(self, ref: WindowRef, pane: Pane) -> bool:
        self.calls.append(("focus", pane.id))
        if self.fail is not None:
            raise self.fail
        return self._focus_ok

    def close(self) -> None:
        """Records the name of the thread it was closed on."""
        self.closed_on.append(threading.current_thread().name)


def two_panes(
    handle: Any = 7,
    *,
    provider: str = "fake",
    left: Rect = Rect(0, 0, 960, 1080),  # noqa: B008 - Rect is immutable
    right: Rect = Rect(960, 0, 960, 1080),  # noqa: B008
    focused: str = "%0",
    taken_at: float = 0.0,
) -> PaneSnapshot:
    return PaneSnapshot(
        handle,
        (
            Pane("%0", left, focused == "%0", provider),
            Pane("%1", right, focused == "%1", provider),
        ),
        taken_at,
    )


__all__ = ["FakeRunner", "PaneError", "RecordingProvider", "two_panes"]
