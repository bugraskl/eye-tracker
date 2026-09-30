"""When to let go of the camera so other apps can use it.

Most webcams can only be opened by one process at a time. Tracking pauses and
releases the camera while another app is streaming from it (video calls) or
while an app from the user's list is running (screen recorders, games).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_SUFFIXES = (".exe", ".app")
_PATH_SEPARATORS = re.compile(r"[\\/]")


@dataclass
class YieldInputs:
    """Facts gathered by the controller for :func:`should_yield`."""

    #: Whether another process streams from a camera (``None`` = cannot tell).
    camera_in_use_by_other: bool | None
    #: Names of running processes (as from ``PlatformServices.running_process_names``).
    running: set[str] = field(default_factory=set)
    #: User-configured process names that pause tracking while they run.
    pause_for_apps: list[str] = field(default_factory=list)


def normalize_process_name(name: str) -> str:
    """Canonical form for matching: lower case, no directory, no ``.exe``/``.app`` suffix."""
    base = _PATH_SEPARATORS.split(name.strip())[-1].strip().lower()
    for suffix in _SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            return base[: -len(suffix)]
    return base


def matching_app(running: Iterable[str], pause_for_apps: Iterable[str]) -> str | None:
    """First entry of ``pause_for_apps`` (as the user wrote it) that is running."""
    names = {normalize_process_name(n) for n in running if n}
    names.discard("")
    for entry in pause_for_apps:
        wanted = normalize_process_name(entry)
        if wanted and wanted in names:
            return entry.strip()
    return None


def should_yield(inputs: YieldInputs, yield_camera: bool) -> tuple[bool, str]:
    """Decide whether tracking should release the camera.

    Returns ``(True, reason)`` - e.g. ``(True, "zoom.exe is running")`` - or
    ``(False, "")``. Listed apps always pause tracking; a camera used by another
    app only does when ``yield_camera`` is enabled.
    """
    app = matching_app(inputs.running or (), inputs.pause_for_apps or ())
    if app is not None:
        return True, f"{app} is running"
    if yield_camera and inputs.camera_in_use_by_other is True:
        return True, "Another app is using the camera"
    return False, ""
