"""Import MediaPipe without its unused heavyweight dependencies.

MediaPipe 1.x imports ``matplotlib.pyplot`` at package import time (only for its
drawing helpers, which this app never calls). Pulling in matplotlib would add
tens of megabytes to every build, so when it is not installed a tiny placeholder
module is registered before MediaPipe is imported. The placeholder raises a
clear error if anything ever tries to use it.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from typing import Any


class _Unavailable(types.ModuleType):
    def __getattr__(self, name: str) -> Any:
        # AttributeError (not RuntimeError) so getattr(mod, x, default) and
        # hasattr() keep working for code that scans sys.modules.
        raise AttributeError(
            f"matplotlib.{name} is unavailable: matplotlib is not bundled with Eye Tracker"
        )


def install() -> None:
    """Register placeholder modules if matplotlib is missing. Idempotent."""
    if "matplotlib" in sys.modules:
        return
    if importlib.util.find_spec("matplotlib") is not None:
        return
    root = _Unavailable("matplotlib")
    pyplot = _Unavailable("matplotlib.pyplot")
    root.pyplot = pyplot  # type: ignore[attr-defined]
    # A real spec keeps importlib.util.find_spec("matplotlib") working afterwards.
    root.__spec__ = importlib.machinery.ModuleSpec("matplotlib", None, is_package=True)
    pyplot.__spec__ = importlib.machinery.ModuleSpec("matplotlib.pyplot", None)
    sys.modules["matplotlib"] = root
    sys.modules["matplotlib.pyplot"] = pyplot


def import_mediapipe() -> Any:
    """Import and return the ``mediapipe`` package (raises ImportError if absent)."""
    install()
    import mediapipe

    return mediapipe
