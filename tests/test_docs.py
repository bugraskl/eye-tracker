"""Documentation stays in sync with the code."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

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


def test_every_setting_is_documented() -> None:
    from eye_tracker.config import describe_settings

    undocumented = [row["key"] for row in describe_settings() if not str(row["doc"]).strip()]
    assert undocumented == []


def test_relative_links_in_markdown_resolve() -> None:
    pages = [
        ROOT / "README.md",
        ROOT / "README.tr.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
    ]
    pages += sorted((ROOT / "docs").glob("*.md"))
    broken = []
    for page in pages:
        for target in re.findall(r"\]\(([^)\s]+)\)", page.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            path = target.split("#", 1)[0]
            if path and not (page.parent / path).exists():
                broken.append(f"{page.relative_to(ROOT)} -> {target}")
    assert broken == []
