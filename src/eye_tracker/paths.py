"""Filesystem locations (config, data, logs, bundled resources)."""

from __future__ import annotations

import getpass
import hashlib
import os
import sys
from pathlib import Path

from platformdirs import PlatformDirs

from . import APP_AUTHOR, APP_SLUG

_override: Path | None = None


def set_base_override(path: Path | None) -> None:
    """Put config, data and logs under one directory (``--config-dir``, tests)."""
    global _override  # noqa: PLW0603 - process-wide override set once at startup / by tests
    _override = Path(path).expanduser().resolve() if path else None


def base_override() -> Path | None:
    """The directory set by :func:`set_base_override` (resolved), or ``None``.

    ``None`` means the default per-user profile. Autostart entries and the
    diagnostics report use it to tell which profile this process runs.
    """
    return _override


def _dirs() -> PlatformDirs:
    return PlatformDirs(appname=APP_SLUG, appauthor=APP_AUTHOR, roaming=False)


def config_dir() -> Path:
    path = _override if _override else Path(_dirs().user_config_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    path = (_override / "data") if _override else Path(_dirs().user_data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_dir() -> Path:
    path = (_override / "logs") if _override else Path(_dirs().user_log_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_file() -> Path:
    return config_dir() / "settings.json"


def calibration_file() -> Path:
    return data_dir() / "calibration.json"


def state_file() -> Path:
    """What the app remembers between runs that is no setting (privacy mode)."""
    return data_dir() / "state.json"


def log_file() -> Path:
    return log_dir() / f"{APP_SLUG}.log"


def ipc_name() -> str:
    """Per-user name for the single-instance local socket."""
    try:
        user = getpass.getuser()
    except Exception:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    scope = f"{user}|{_override}" if _override else user
    digest = hashlib.sha1(scope.encode("utf-8")).hexdigest()[:10]
    return f"{APP_SLUG}-{digest}"


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def package_dir() -> Path:
    """Directory of the ``eye_tracker`` package (works for source and frozen builds)."""
    if is_frozen():
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        return base / "eye_tracker"
    return Path(__file__).resolve().parent


def model_path(filename: str) -> Path:
    return package_dir() / "vision" / "models" / filename
