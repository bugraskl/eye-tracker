#!/usr/bin/env python3
"""Render the application icon into every format the packages need.

The artwork is drawn in code by ``eye_tracker.ui.icons.render_app_icon_png``, so
the app, the tray and the installers share one source of truth. This script
renders it headless (Qt's ``offscreen`` platform) at each native size, so small
icons stay crisp instead of being scaled down from 1024 px, and writes:

=====================================  ===============  ==============================
File                                   Sizes            Used by
=====================================  ===============  ==============================
``assets/icon.png``                    1024             README, release pages
``packaging/windows/eye-tracker.ico``  16 - 256         executables, installer
``packaging/macos/eye-tracker.icns``   32 - 1024       ``Eye Tracker.app``
``packaging/linux/eye-tracker.png``    512              AppImage, ``.desktop`` entry
=====================================  ===============  ==============================

The ``.icns`` uses the macOS icon grid (a smaller tile with a soft shadow, the
``padded`` variant); the other formats use the full square.

Usage::

    uv run python scripts/make_icons.py            # write all icons
    uv run python scripts/make_icons.py --out DIR  # write under DIR instead of the repo
    uv run python scripts/make_icons.py --check    # exit 1 if any icon file is missing

Pillow (``build`` dependency group) assembles the ``.ico`` and ``.icns`` containers.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from PySide6.QtGui import QImage

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Sizes stored in the Windows .ico (Explorer, taskbar, Alt+Tab, 100-200 % DPI).
ICO_SIZES: tuple[int, ...] = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
#: Pixel sizes Pillow writes into an .icns (ic07..ic14: 16-512 pt at 1x and 2x).
ICNS_SIZES: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)

OUTPUTS: dict[str, Path] = {
    "png": Path("assets/icon.png"),
    "ico": Path("packaging/windows/eye-tracker.ico"),
    "icns": Path("packaging/macos/eye-tracker.icns"),
    "linux": Path("packaging/linux/eye-tracker.png"),
}

#: ``render(size, padded)`` -> a ``size`` x ``size`` QImage.
Renderer = Callable[[int, bool], "QImage"]


def _to_pil(image: QImage) -> PILImage:
    """Convert a QImage to an RGBA Pillow image without re-encoding it."""
    from PIL import Image
    from PySide6.QtGui import QImage as _QImage

    # RGBA8888 is straight (non-premultiplied) alpha in byte order R, G, B, A,
    # which is exactly Pillow's "RGBA" raw layout.
    rgba = image.convertToFormat(_QImage.Format.Format_RGBA8888)
    size = (rgba.width(), rgba.height())
    data = bytes(rgba.constBits())
    return Image.frombuffer("RGBA", size, data, "raw", "RGBA", rgba.bytesPerLine(), 1).copy()


def _render(render: Renderer, size: int, *, padded: bool = False) -> PILImage:
    image = _to_pil(render(size, padded))
    if image.size != (size, size):
        raise RuntimeError(f"renderer returned {image.size} for a {size} px icon")
    return image


def write_icons(render: Renderer, root: Path) -> list[Path]:
    """Render every icon file below ``root``; returns the paths written."""
    full = {size: _render(render, size) for size in sorted({*ICO_SIZES, 512, 1024})}
    padded = {size: _render(render, size, padded=True) for size in ICNS_SIZES}
    written: list[Path] = []

    def target(key: str) -> Path:
        path = root / OUTPUTS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        written.append(path)
        return path

    full[1024].save(target("png"), format="PNG", optimize=True)
    full[512].save(target("linux"), format="PNG", optimize=True)

    # Every size is supplied explicitly; Pillow would otherwise downscale the
    # largest image, which blurs the 16-32 px variants.
    ico_frames = [full[size] for size in ICO_SIZES]
    ico_frames[-1].save(
        target("ico"),
        format="ICO",
        sizes=[(size, size) for size in ICO_SIZES],
        append_images=ico_frames[:-1],
    )
    padded[1024].save(
        target("icns"),
        format="ICNS",
        append_images=[padded[size] for size in ICNS_SIZES if size != 1024],
    )
    return written


def missing_outputs(root: Path) -> list[Path]:
    """Icon files that do not exist below ``root``."""
    return [root / rel for rel in OUTPUTS.values() if not (root / rel).is_file()]


def _app_renderer() -> tuple[Renderer, object]:
    """The app's icon renderer plus the Qt application object it needs.

    The caller must keep the application object alive while rendering. Raises
    ImportError when Qt or the icon module is unavailable.
    """
    # Must be set before the first Qt import so no display is needed (CI, SSH).
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from PySide6.QtGui import QGuiApplication

    from eye_tracker.ui import icons

    app = QGuiApplication.instance() or QGuiApplication(["make_icons"])
    render_png = getattr(icons, "render_app_icon_png", None)
    if render_png is None:
        raise ImportError("eye_tracker.ui.icons has no render_app_icon_png()")
    supports_padding = "padded" in inspect.signature(render_png).parameters

    def render(size: int, padded: bool) -> QImage:
        if padded and supports_padding:
            return render_png(size, padded=True)
        return render_png(size)

    return render, app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render Eye Tracker's icons for all platforms.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT,
        help="root directory for the output files (default: the repository)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not render; exit 1 if any icon file is missing",
    )
    args = parser.parse_args(argv)
    root: Path = args.out

    if args.check:
        missing = missing_outputs(root)
        for path in missing:
            print(f"  missing  {path}")
        if missing:
            print("make_icons: run `uv run python scripts/make_icons.py`", file=sys.stderr)
            return 1
        print(f"make_icons: OK: all {len(OUTPUTS)} icon files present")
        return 0

    try:
        renderer, _app = _app_renderer()
    except ImportError as exc:
        print(f"make_icons: cannot import the icon renderer: {exc}", file=sys.stderr)
        print("make_icons: run `uv sync` first (Pillow is in the 'build' group)", file=sys.stderr)
        return 1
    for path in write_icons(renderer, root):
        shown = path.relative_to(root).as_posix() if path.is_relative_to(root) else path
        print(f"  wrote {shown}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
