"""Documentation stays in sync with the code."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen_config_docs", ROOT / "scripts" / "gen_config_docs.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configuration_reference_is_up_to_date() -> None:
    generator = _load_generator()
    assert generator.main(["--check"]) == 0, "run: uv run python scripts/gen_config_docs.py"


def test_hotkey_defaults_are_listed_per_platform_only_when_they_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _load_generator()
    config = generator.config
    monkeypatch.setattr(config, "_HOTKEY_MODIFIERS", {"win32": "ctrl+alt+meta"})
    monkeypatch.setattr(config, "_HOTKEY_MODIFIERS_OTHER", "ctrl+alt+meta")
    assert generator._default("hotkeys.toggle_tracking", "ctrl+alt+meta+t") == "`ctrl+alt+meta+t`"
    monkeypatch.setattr(config, "_HOTKEY_MODIFIERS", {"darwin": "ctrl+alt"})
    assert generator._default("hotkeys.recalibrate", "ctrl+alt+meta+c") == (
        "`ctrl+alt+meta+c` Windows<br>`ctrl+alt+c` macOS<br>`ctrl+alt+meta+c` Linux"
    )
    # Not a hotkey: as it is.
    assert generator._default("camera.device", "0") == "`0`"


def test_section_notes_take_their_numbers_from_the_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The countdown is part of the away time (r2-docs-06): the page says when it starts."""
    generator = _load_generator()
    monkeypatch.setattr(
        generator.config,
        "PresenceSettings",
        lambda: SimpleNamespace(away_timeout_s=60, warning_s=15),
    )
    notes = generator.section_notes()
    assert "appears after 45 s without you, and the action follows at 60 s" in notes["presence"]
    assert f"`{generator.config.HotkeySettings().toggle_tracking}`" in notes["hotkeys"]


def test_every_setting_is_documented() -> None:
    from eye_tracker.config import describe_settings

    undocumented = [row["key"] for row in describe_settings() if not str(row["doc"]).strip()]
    assert undocumented == []


PAGES = [
    ROOT / "README.md",
    ROOT / "README.tr.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "CHANGELOG.md",
    *sorted((ROOT / "docs").glob("*.md")),
]


def _local_targets(page: Path) -> list[str]:
    """Relative link and image targets of a Markdown page: ``[text](target)`` and the
    ``<img src="target">`` of the HTML blocks (the README hero and demo)."""
    text = page.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r"\ssrc=\"([^\"]+)\"", text)
    return [t for t in targets if "://" not in t and not t.startswith(("#", "mailto:"))]


def test_relative_links_in_markdown_resolve() -> None:
    broken = []
    for page in PAGES:
        for target in _local_targets(page):
            path = target.split("#", 1)[0]
            if path and not (page.parent / path).exists():
                broken.append(f"{page.relative_to(ROOT)} -> {target}")
    # assets/hero.png is rendered from assets/hero.svg and committed with it.
    assert broken == [], f"missing link or image targets: {broken}"


def test_images_are_checked_too(tmp_path: Path) -> None:
    """Regression (r2-docs-11): a missing README hero image went unnoticed, because
    only Markdown links were checked, not ``<img src>``."""
    page = tmp_path / "README.md"
    page.write_text(
        '<img src="assets/hero.png" alt="x">\n[docs](docs/a.md) ![shot](shot.png)\n'
        '<img src="https://example.org/badge.svg">\n',
        encoding="utf-8",
    )
    assert _local_targets(page) == ["docs/a.md", "shot.png", "assets/hero.png"]
