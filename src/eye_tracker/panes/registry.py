"""Which pane providers look at which windows.

Before any provider sees a window, :func:`is_denied` checks it against a hard
deny-list of Electron and Chromium applications (editors, browsers, chat apps).
Asking those about their contents through accessibility interfaces can switch
them into a slower screen-reader mode, so they are never inspected at all, not
even by a provider's :meth:`~.types.PaneProvider.applies`. Editors such as VS
Code get their own, purpose-built support later instead.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from ..types import AppIdentity, Rect, WindowRef
from .types import PaneProvider

if TYPE_CHECKING:
    from ..config import PaneSettings
    from .providers.command import CommandRunner

log = logging.getLogger(__name__)

#: Process names (lower-cased, without extension) that are never inspected.
DENIED_PROCESSES: frozenset[str] = frozenset(
    {
        "code",
        "code - insiders",
        "cursor",
        "claude",
        "chatgpt",
        "chrome",
        "google chrome",
        "chromium",
        "msedge",
        "microsoft edge",
        "firefox",
        "slack",
        "teams",
        "ms-teams",
        "discord",
        "obsidian",
        "electron",
    }
)

#: Window classes (Windows), WM_CLASS classes (X11) and bundle ids (macOS) that
#: are never inspected. ``Chrome_WidgetWin_1`` is the top-level window class of
#: every Chromium and Electron application on Windows.
DENIED_APP_IDS: frozenset[str] = frozenset(
    {
        "Chrome_WidgetWin_1",
        "com.microsoft.VSCode",
        "com.microsoft.VSCodeInsiders",
        "com.todesktop.230313mzl4w4u92",  # Cursor
        "com.anthropic.claudefordesktop",
        "com.openai.chat",
        "com.google.Chrome",
        "com.microsoft.edgemac",
        "org.mozilla.firefox",
        "com.tinyspeck.slackmacgap",
        "com.microsoft.teams2",
        "com.hnc.Discord",
        "md.obsidian",
        "com.github.Electron",
    }
)


def is_denied(app: AppIdentity) -> bool:
    """Whether windows of ``app`` must never be inspected for panes."""
    return app.process.strip().lower() in DENIED_PROCESSES or app.app_id in DENIED_APP_IDS


class PaneRegistry:
    """The enabled providers, in order of preference."""

    def __init__(self, providers: Iterable[PaneProvider] = ()) -> None:
        self._providers: tuple[PaneProvider, ...] = tuple(providers)

    @property
    def providers(self) -> tuple[PaneProvider, ...]:
        return self._providers

    def providers_for(self, app: AppIdentity | None) -> list[PaneProvider]:
        """The providers that may know panes of ``app``; none for a denied app.

        The deny-list is checked before any provider is asked anything.
        """
        if app is None or is_denied(app):
            return []
        found: list[PaneProvider] = []
        for provider in self._providers:
            try:
                if provider.applies(app):
                    found.append(provider)
            except Exception:
                log.debug("Pane provider %s failed in applies()", provider.name, exc_info=True)
        return found

    def provider(self, name: str) -> PaneProvider | None:
        return next((p for p in self._providers if p.name == name), None)


#: Optional providers implemented in their own module: (setting, module, class).
#: A slot whose module does not exist yet is skipped. The class is created with
#: ``client_rect=`` and ``runner=`` like the built-in ones.
_OPTIONAL_PROVIDERS: tuple[tuple[str, str, str], ...] = (
    ("windows_terminal", "windows_terminal", "WindowsTerminalProvider"),
)


def default_providers(
    settings: PaneSettings,
    *,
    client_rect: Callable[[WindowRef], Rect | None],
    runner: CommandRunner | None = None,
) -> list[PaneProvider]:
    """The providers the settings enable, most specific first.

    ``client_rect`` is ``PlatformServices.window_client_rect`` (panes are mapped
    onto the content area); ``runner`` runs the terminals' command-line tools
    (tests pass a fake).
    """
    from .providers.command import run_command
    from .providers.tmux import TmuxProvider
    from .providers.wezterm import WezTermProvider

    run = runner if runner is not None else run_command
    providers: list[PaneProvider] = []
    if settings.wezterm:
        # Before tmux: inside WezTerm its own panes are what the user sees split.
        providers.append(WezTermProvider(client_rect=client_rect, runner=run))
    for setting, module_name, class_name in _OPTIONAL_PROVIDERS:
        if not getattr(settings, setting, False):
            continue
        provider = _optional_provider(module_name, class_name, client_rect, run)
        if provider is not None:
            providers.append(provider)
    if settings.tmux:
        providers.append(TmuxProvider(client_rect=client_rect, runner=run))
    return providers


def _optional_provider(
    module_name: str,
    class_name: str,
    client_rect: Callable[[WindowRef], Rect | None],
    runner: CommandRunner,
) -> PaneProvider | None:
    try:
        module = importlib.import_module(f"{__package__}.providers.{module_name}")
    except ImportError:
        return None  # not implemented yet
    try:
        cls = getattr(module, class_name)
        provider: PaneProvider = cls(client_rect=client_rect, runner=runner)
    except Exception:
        log.warning("Pane provider %s could not be created", class_name, exc_info=True)
        return None
    return provider
