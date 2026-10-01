"""Entry point for the frozen (PyInstaller) builds of Eye Tracker.

Every executable in a bundle is built from this one script, so it chooses the
entry point from the name it was started as:

* ``EyeTracker.exe`` (Windows) and ``Eye Tracker`` (inside the macOS app) are the
  windowed tray app and call :func:`eye_tracker.cli.gui_main`.
* Anything else, i.e. ``eye-tracker-cli`` (Windows, macOS), ``eye-tracker.exe``
  (the copy of ``eye-tracker-cli.exe`` that the Windows installer puts on PATH)
  and ``eye-tracker`` (Linux, where one executable serves both roles), calls
  :func:`eye_tracker.cli.main`.

The names are set in ``eye-tracker.spec``; keep ``_GUI_EXECUTABLES`` in sync with it.
"""

from __future__ import annotations

import os
import sys

#: Case-folded stems of the windowed executables.
_GUI_EXECUTABLES = frozenset({"eyetracker", "eye tracker"})


def _is_gui_executable(path: str) -> bool:
    """Whether ``path`` names one of the windowed executables.

    Both separators are accepted on every OS: ``os.path`` is ``posixpath`` on
    macOS and Linux and would treat a Windows path as a single file name. The
    real ``sys.executable`` is always native; this keeps the check (and its
    tests) independent of the platform it runs on.
    """
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    stem = os.path.splitext(name)[0]
    return stem.casefold() in _GUI_EXECUTABLES


def _run() -> int | None:
    from eye_tracker import cli

    if _is_gui_executable(sys.executable):
        return cli.gui_main()
    return cli.main()


if __name__ == "__main__":
    sys.exit(_run())
