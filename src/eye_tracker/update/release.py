"""Versions and the shape of a GitHub release, without any networking.

The update check asks GitHub for the latest release (``releases/latest``, which
never answers with a draft or a pre-release) and keeps only what it can trust:

* the tag must be a plain ``X.Y.Z`` version;
* the page of the release is built here, not taken from the answer;
* an asset counts only when its download address is exactly the one GitHub
  derives from the repository, the tag and the file name, so a crafted answer
  cannot point the installer download somewhere else.

Only the Windows installer is installed by the app (see :mod:`.installer`); the
other packages are installed by hand from the release page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .. import REPO_URL

REPO_SLUG = REPO_URL.removeprefix("https://github.com/")
#: The one address the update check asks.
LATEST_RELEASE_URL = f"https://api.github.com/repos/{REPO_SLUG}/releases/latest"
RELEASES_PAGE = f"{REPO_URL}/releases"

#: The file with the SHA-256 of every package of a release (``release.yml``).
CHECKSUMS_NAME = "SHA256SUMS.txt"
#: Largest installer the app will download (the real one is about 150 MB).
MAX_INSTALLER_BYTES = 400 * 1024 * 1024
MAX_CHECKSUMS_BYTES = 64 * 1024
#: Largest answer of the release API read.
MAX_API_BYTES = 1024 * 1024

Version = tuple[int, int, int]

_STABLE = re.compile(r"^v?(\d{1,4})\.(\d{1,4})\.(\d{1,4})$")
_LEADING = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


def parse_stable(text: object) -> Version | None:
    """``"0.2.1"`` or ``"v0.2.1"`` as a tuple; anything else (rc, dev, junk) is ``None``."""
    if not isinstance(text, str):
        return None
    match = _STABLE.match(text.strip())
    return None if match is None else (int(match[1]), int(match[2]), int(match[3]))


def parse_running(text: str) -> Version | None:
    """The version of this build: its leading numbers (``0.3.0.dev1`` is 0.3.0)."""
    match = _LEADING.match(text.strip())
    return None if match is None else (int(match[1]), int(match[2]), int(match[3]))


def format_version(version: Version) -> str:
    return ".".join(str(part) for part in version)


def installer_name(version: str) -> str:
    """File name of the Windows installer of ``version`` (see ``installer.iss``)."""
    return f"EyeTracker-{version}-windows-x64-setup.exe"


def download_url(tag: str, name: str) -> str:
    """Where GitHub serves the asset ``name`` of the release ``tag``."""
    return f"{REPO_URL}/releases/download/{tag}/{name}"


@dataclass(frozen=True, slots=True)
class Asset:
    name: str
    url: str
    size: int


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    """A published release: its version, its page and the assets the app uses."""

    version: str
    tag: str
    page_url: str
    installer: Asset | None
    checksums: Asset | None

    @property
    def version_tuple(self) -> Version:
        parsed = parse_stable(self.version)
        assert parsed is not None  # parse_release only builds stable versions
        return parsed

    @property
    def installable(self) -> bool:
        """Whether the app can install this release by itself (installer and checksums)."""
        return self.installer is not None and self.checksums is not None


def _asset(payload: Any, tag: str, name: str, limit: int) -> Asset | None:
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, dict) or item.get("name") != name:
            continue
        size = item.get("size")
        url = item.get("browser_download_url")
        if (
            isinstance(size, int)
            and not isinstance(size, bool)
            and 0 < size <= limit
            and url == download_url(tag, name)
        ):
            return Asset(name, url, size)
    return None


def parse_release(payload: object) -> ReleaseInfo | None:
    """The release described by a ``releases/latest`` answer; ``None`` if it is not
    a usable stable release."""
    if not isinstance(payload, dict) or payload.get("draft") or payload.get("prerelease"):
        return None
    tag = payload.get("tag_name")
    version = parse_stable(tag)
    if version is None or not isinstance(tag, str):
        return None
    tag = tag.strip()
    text = format_version(version)
    assets = payload.get("assets")
    return ReleaseInfo(
        version=text,
        tag=tag,
        page_url=f"{REPO_URL}/releases/tag/{tag}",
        installer=_asset(assets, tag, installer_name(text), MAX_INSTALLER_BYTES),
        checksums=_asset(assets, tag, CHECKSUMS_NAME, MAX_CHECKSUMS_BYTES),
    )


def is_newer(release: ReleaseInfo, running: str) -> bool:
    """Whether ``release`` is newer than the running build ``running``."""
    current = parse_running(running)
    return current is not None and release.version_tuple > current
