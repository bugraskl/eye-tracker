"""Build the project website (GitHub Pages) from site/ into an output folder.

Usage:
    uv run python scripts/build_site.py --release release.json --out _site
    uv run python scripts/build_site.py --out _site              # no release data

The HTML pages under site/ contain ``{{PLACEHOLDER}}`` markers for the latest
release (version, date, download links and sizes). ``--release`` takes the JSON
of GitHub's "get the latest release" API (``gh api repos/OWNER/REPO/releases/latest``).
Without it, or when an asset is missing, links fall back to the releases page so
the site is always deployable. Images are copied from the repository's assets/
folder, which stays the single source.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
REPO_URL = "https://github.com/bugraskl/eye-tracker"
SITE_URL = "https://bugraskl.github.io/eye-tracker/"
RELEASES_URL = f"{REPO_URL}/releases/latest"

#: Placeholder prefix → regular expression for the release asset's file name.
ASSETS: dict[str, str] = {
    "WIN_SETUP": r"-windows-x64-setup\.exe$",
    "WIN_ZIP": r"-windows-x64-portable\.zip$",
    "MAC_DMG": r"-macos-arm64\.dmg$",
    "LINUX_APPIMAGE": r"-linux-x86_64\.AppImage$",
    "LINUX_TAR": r"-linux-x86_64\.tar\.gz$",
    "SHA256SUMS": r"^SHA256SUMS\.txt$",
}

#: Files copied from the repository's assets/ folder: source → published name.
REPO_ASSETS = {
    "demo.svg": "demo.svg",
    "hero.png": "og.png",
}

#: The demo animation's captions for the Turkish page (assets/demo-tr.svg).
DEMO_CAPTIONS_TR = {
    "Looking right → cursor and keyboard focus on the right screen": (
        "Sağa bakınca → imleç ve klavye odağı sağ ekranda"
    ),
    "Looking left → cursor and keyboard focus jump to the left screen": (
        "Sola bakınca → imleç ve klavye odağı sol ekrana geçer"
    ),
}

MONTHS = {
    "en": "January February March April May June July August September October November December",
    "tr": "Ocak Şubat Mart Nisan Mayıs Haziran Temmuz Ağustos Eylül Ekim Kasım Aralık",
}

PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def human_size(size: int) -> str:
    """``63204271`` → ``"63 MB"`` (decimal units, like most download pages)."""
    if size <= 0:
        return ""
    if size >= 1_000_000:
        return f"{size / 1_000_000:.0f} MB"
    return f"{max(size / 1000, 1):.0f} kB"


def human_date(value: str, lang: str) -> str:
    try:
        day = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return ""
    month = MONTHS[lang].split()[day.month - 1]
    return f"{day.day} {month} {day.year}" if lang == "tr" else f"{month} {day.day}, {day.year}"


def release_values(release: dict[str, Any], lang: str) -> dict[str, str]:
    """Placeholder values for one language from the release JSON (may be empty)."""
    tag = str(release.get("tag_name") or "")
    values = {
        "SITE_URL": SITE_URL,
        "VERSION": tag.removeprefix("v") or "latest",
        "RELEASE_URL": str(release.get("html_url") or RELEASES_URL),
        "RELEASE_DATE": human_date(str(release.get("published_at") or ""), lang),
        "YEAR": str(date.today().year),
    }
    assets = [a for a in release.get("assets") or [] if isinstance(a, dict)]
    for key, pattern in ASSETS.items():
        match = next((a for a in assets if re.search(pattern, str(a.get("name", "")))), None)
        values[f"{key}_URL"] = str(match["browser_download_url"]) if match else RELEASES_URL
        values[f"{key}_SIZE"] = human_size(int(match.get("size") or 0)) if match else ""
    return values


def render(text: str, values: dict[str, str], source: Path) -> str:
    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise SystemExit(f"{source}: unknown placeholder {{{{{key}}}}}")
        return values[key]

    return PLACEHOLDER.sub(substitute, text)


def page_language(path: Path) -> str:
    return "tr" if path.relative_to(SITE).parts[0] == "tr" else "en"


def build(release: dict[str, Any], out: Path) -> list[Path]:
    """Write the site into ``out`` (emptied first) and return the pages written."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pages: list[Path] = []
    for source in sorted(SITE.rglob("*")):
        if source.is_dir():
            continue
        target = out / source.relative_to(SITE)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".html":
            html = render(
                source.read_text(encoding="utf-8"),
                release_values(release, page_language(source)),
                source,
            )
            target.write_text(html, encoding="utf-8", newline="\n")
            pages.append(target)
        else:
            shutil.copy2(source, target)
    for name, published in REPO_ASSETS.items():
        shutil.copy2(ROOT / "assets" / name, out / "assets" / published)
    _write_turkish_demo(out / "assets" / "demo-tr.svg")
    _write_icons(out / "assets")
    # GitHub Pages: serve files as they are (no Jekyll processing).
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return pages


def _write_turkish_demo(target: Path) -> None:
    """The demo animation with Turkish captions (the English file stays the source)."""
    svg = (ROOT / "assets" / "demo.svg").read_text(encoding="utf-8")
    for english, turkish in DEMO_CAPTIONS_TR.items():
        if english not in svg:
            raise SystemExit(f"assets/demo.svg: caption not found: {english!r}")
        svg = svg.replace(english, turkish)
    target.write_text(svg, encoding="utf-8", newline="\n")


def _write_icons(folder: Path) -> None:
    """Favicons resized from the app icon (assets/icon.png, 1024 px)."""
    try:
        from PIL import Image
    except ImportError:  # Pillow is a dev dependency; CI installs it.
        shutil.copy2(ROOT / "assets" / "icon.png", folder / "favicon.png")
        shutil.copy2(ROOT / "assets" / "icon.png", folder / "apple-touch-icon.png")
        return
    with Image.open(ROOT / "assets" / "icon.png") as icon:
        icon.convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS).save(
            folder / "favicon.png", optimize=True
        )
        icon.convert("RGBA").resize((180, 180), Image.Resampling.LANCZOS).save(
            folder / "apple-touch-icon.png", optimize=True
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, help="JSON of the latest GitHub release")
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="output folder")
    args = parser.parse_args(argv)
    release: dict[str, Any] = {}
    if args.release and args.release.is_file():
        try:
            loaded = json.loads(args.release.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as exc:
            print(f"build_site: ignoring unreadable release data: {exc}", file=sys.stderr)
        else:
            release = loaded if isinstance(loaded, dict) else {}
    pages = build(release, args.out)
    version = release.get("tag_name") or "no release data"
    print(f"build_site: {len(pages)} pages ({version}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
