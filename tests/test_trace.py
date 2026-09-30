"""The opt-in tracking trace writes numbers only, one JSON object per line."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from eye_tracker import trace as trace_mod
from eye_tracker.trace import TraceWriter


def test_writes_json_lines_and_rounds(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "trace.jsonl"
    writer = TraceWriter(path)
    writer.write({"t": writer.number(1.23456789, 3), "f": writer.vector(np.array([0.1, math.nan]))})
    writer.write({"t": 2.0, "f": None})
    writer.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"t": 1.235, "f": [0.1, None]},
        {"t": 2.0, "f": None},
    ]


def test_line_cap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(trace_mod, "MAX_LINES", 3)
    writer = TraceWriter(tmp_path / "t.jsonl")
    for i in range(10):
        writer.write({"i": i})
    writer.close()
    assert len((tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_unwritable_path_disables_quietly(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    writer = TraceWriter(blocker / "trace.jsonl")  # parent is a file
    writer.write({"a": 1})
    writer.write({"a": 2})
    writer.close()  # no exception
    assert not (blocker / "trace.jsonl").exists()


def test_helpers_handle_bad_values() -> None:
    assert TraceWriter.number(None) is None
    assert TraceWriter.number("x") is None
    assert TraceWriter.number(math.inf) is None
    assert TraceWriter.vector(None) is None
    assert TraceWriter.vector(5) is None
