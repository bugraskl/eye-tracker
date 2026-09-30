"""Entry point for the frozen (PyInstaller) builds of Eye Tracker.

Every executable in a bundle is built from this one script, so it chooses the
entry point from the name it was started as:

* ``EyeTracker.exe`` (Windows) and ``Eye Tracker`` (inside the macOS app) are the
  windowed tray app and call :func:`eye_tracker.cli.gui_main`.
* Anything else, i.e. ``eye-tracker-cli`` (Windows, macOS) and ``eye-tracker``
  (Linux, where one executable serves both roles), calls :func:`eye_tracker.cli.main`.

The names are set in ``eye-tracker.spec``; keep ``_GUI_EXECUTABLES`` in sync with it.
"""

from __future__ import annotations

import os
import sys

#: Case-folded stems of the windowed executables.
_GUI_EXECUTABLES = frozenset({"eyetracker", "eye tracker"})


def _is_gui_executable(path: str) -> bool:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.casefold() in _GUI_EXECUTABLES


def _run() -> int | None:
    from eye_tracker import cli

    if _is_gui_executable(sys.executable):
        return cli.gui_main()
    return cli.main()


if __name__ == "__main__":
    sys.exit(_run())
