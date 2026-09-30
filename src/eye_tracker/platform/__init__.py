"""Operating-system integration.

Use :func:`get_platform` to obtain the implementation for the running OS. It
never fails: if the OS-specific module cannot be imported (a missing optional
dependency, an unknown OS) the do-nothing :class:`PlatformServices` is returned
and the app keeps working with reduced features.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading

from .base import PlatformServices

log = logging.getLogger(__name__)

__all__ = ["PlatformServices", "get_platform"]

#: ``sys.platform`` prefix -> (module, class) of the implementation.
_IMPLEMENTATIONS: dict[str, tuple[str, str]] = {
    "win32": ("windows", "WindowsPlatform"),
    "darwin": ("macos", "MacPlatform"),
    "linux": ("linux", "LinuxPlatform"),
}

_lock = threading.Lock()
_cache: dict[str, PlatformServices] = {}


def get_platform() -> PlatformServices:
    """The process-wide platform services singleton (created on first call)."""
    with _lock:
        instance = _cache.get("instance")
        if instance is None:
            instance = _create(sys.platform)
            _cache["instance"] = instance
            log.debug("Platform services: %s", type(instance).__name__)
        return instance


def _create(system: str) -> PlatformServices:
    for prefix, (module_name, class_name) in _IMPLEMENTATIONS.items():
        if not system.startswith(prefix):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
            cls = getattr(module, class_name)
            instance = cls()
        except Exception as exc:
            log.warning("%s integration unavailable (%s); OS features are disabled", system, exc)
            return PlatformServices()
        if not isinstance(instance, PlatformServices):
            log.warning("%s.%s is not a PlatformServices; ignored", module_name, class_name)
            return PlatformServices()
        return instance
    log.info("No OS integration for %r; OS features are disabled", system)
    return PlatformServices()


def _reset_for_tests() -> None:
    """Forget the cached singleton (tests only)."""
    with _lock:
        _cache.clear()
