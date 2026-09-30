"""``python -m eye_tracker``: the same as the ``eye-tracker`` command."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
