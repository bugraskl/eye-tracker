"""Opt-in tracking trace for tuning and bug reports (``eye-tracker run --trace FILE``).

Each analysed frame becomes one JSON line of *numbers*: the backend's feature
vector, the estimated gaze point, the cursor position and the switching decision.
No image data is ever written. The file is only created when the user asks for
it on the command line, and it stays on the local disk.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import threading
from pathlib import Path
from typing import IO, Any

log = logging.getLogger(__name__)

#: Stop writing after this many lines (~100 MB) so a forgotten trace cannot fill a disk.
MAX_LINES = 500_000


def _num(value: Any, digits: int = 4) -> Any:
    """JSON-safe rounded number (``None`` for NaN/inf/missing)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if math.isfinite(f) else None


class TraceWriter:
    """Appends JSON lines to a file; thread-safe and failure-tolerant."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()
        self._fh: IO[str] | None = None
        self._lines = 0
        self._failed = False

    def write(self, record: dict[str, Any]) -> None:
        """Append one record. Errors are logged once and then ignored."""
        if self._failed or self._lines >= MAX_LINES:
            return
        with self._lock:
            try:
                if self._fh is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._fh = self.path.open("a", encoding="utf-8", newline="\n")
                    log.info("Writing tracking trace to %s", self.path)
                self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
                self._lines += 1
                if self._lines % 50 == 0:
                    self._fh.flush()
                if self._lines == MAX_LINES:
                    log.warning("Trace reached %d lines; no further lines are written", MAX_LINES)
            except (OSError, TypeError, ValueError) as exc:
                self._failed = True
                log.warning("Tracking trace disabled: %s", exc)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                with contextlib.suppress(OSError):
                    self._fh.close()
                self._fh = None

    @staticmethod
    def number(value: Any, digits: int = 4) -> Any:
        return _num(value, digits)

    @staticmethod
    def vector(values: Any, digits: int = 4) -> list[Any] | None:
        if values is None:
            return None
        try:
            return [_num(v, digits) for v in values]
        except TypeError:
            return None
