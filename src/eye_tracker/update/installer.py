"""Downloading, verifying and starting the Windows installer of a newer release.

The app installs an update only on Windows, and only when it was itself
installed by the setup program (it sits next to ``unins000.exe``): the new setup
replaces the files of that very installation, closes the running app and starts
it again, the way ``winget upgrade`` does (see ``installer.iss``). The portable
zip, macOS and Linux get a notice and the address of the release page instead.

What is checked before the setup program is started:

* the file comes from the release's own download address (see
  :mod:`.release`), over HTTPS, from GitHub's hosts only (see :mod:`.fetch`);
* its size is the size GitHub announced;
* its SHA-256 equals the one in the release's ``SHA256SUMS.txt``.

The checksum file lives next to the installer, so it protects against a damaged
or truncated download, not against someone who can replace both files on GitHub.
Releases also carry Sigstore build attestations, which `gh attestation verify`
checks (see ``docs/privacy.md``); the app does not run that check itself.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .. import paths
from .fetch import Cancelled, Fetcher
from .release import MAX_CHECKSUMS_BYTES, ReleaseInfo

log = logging.getLogger(__name__)

#: ``install_kind()`` answers.
KIND_SETUP = "setup"
KIND_PORTABLE = "portable"
KIND_UNSUPPORTED = "unsupported"

_UNINSTALLER = "unins000.exe"
#: Switches of the Inno Setup program: no windows, no prompts, no restart.
SETUP_ARGUMENTS = ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART")
_CHECKSUM_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](\S.*?)\s*$")
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200


class UpdateError(Exception):
    """An update that cannot be installed (the message is meant for the user)."""


def install_kind() -> str:
    """How this copy of the app can be updated: by its setup program, by hand
    (portable zip), or not by the app (other platforms, a source checkout)."""
    if sys.platform != "win32" or not paths.is_frozen():
        return KIND_UNSUPPORTED
    if (Path(sys.executable).resolve().parent / _UNINSTALLER).is_file():
        return KIND_SETUP
    return KIND_PORTABLE


def updates_dir() -> Path:
    return paths.data_dir() / "updates"


def parse_checksums(text: str) -> dict[str, str]:
    """``{file name: lower-case SHA-256}`` from a ``sha256sum`` listing."""
    sums: dict[str, str] = {}
    for line in text.splitlines():
        match = _CHECKSUM_LINE.match(line)
        if match is not None:
            sums[match[2]] = match[1].lower()
    return sums


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_installer(
    release: ReleaseInfo,
    fetcher: Fetcher,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    directory: Path | None = None,
) -> Path:
    """Download the installer of ``release`` and verify it; returns its path.

    Raises :class:`UpdateError` when it cannot be trusted, :class:`Cancelled` when
    ``cancelled()`` turns true, and ``NetworkError`` when the download fails. A
    file that fails a check is deleted.
    """
    installer, checksums = release.installer, release.checksums
    if installer is None or checksums is None:
        raise UpdateError("This release has no installer the app can verify.")
    text = fetcher.get(checksums.url, max_bytes=MAX_CHECKSUMS_BYTES).decode("utf-8", "replace")
    expected = parse_checksums(text).get(installer.name)
    if expected is None:
        raise UpdateError("The release's checksum list does not name the installer.")

    folder = directory if directory is not None else updates_dir()
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / installer.name
    part = folder / (installer.name + ".part")
    for stale in (part, final):
        with contextlib.suppress(OSError):
            stale.unlink()
    try:
        size = fetcher.download(
            installer.url, part, max_bytes=installer.size, progress=progress, cancelled=cancelled
        )
        if size != installer.size:
            raise UpdateError("The download is incomplete.")
        if not hmac.compare_digest(file_sha256(part), expected):
            raise UpdateError("The download does not match its checksum and was discarded.")
        part.replace(final)
    except BaseException:
        with contextlib.suppress(OSError):
            part.unlink()
        raise
    return final


def launch_installer(installer: Path) -> None:
    """Start the setup program in the background. It asks the running app to quit,
    replaces the files and starts the app again; this process does not wait."""
    if install_kind() != KIND_SETUP:
        raise UpdateError("This copy of the app is not installed by its setup program.")
    command = [str(installer), *SETUP_ARGUMENTS]
    log.info("Starting the installer %s", installer.name)
    flags = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    subprocess.Popen(
        command,
        close_fds=True,
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def remove_downloads() -> None:
    """Delete what earlier updates left in the downloads folder (best effort)."""
    folder = updates_dir()
    if not folder.is_dir():
        return
    for item in folder.iterdir():
        with contextlib.suppress(OSError):
            if item.is_file():
                item.unlink()


__all__ = [
    "KIND_PORTABLE",
    "KIND_SETUP",
    "KIND_UNSUPPORTED",
    "SETUP_ARGUMENTS",
    "Cancelled",
    "UpdateError",
    "download_installer",
    "file_sha256",
    "install_kind",
    "launch_installer",
    "parse_checksums",
    "remove_downloads",
    "updates_dir",
]
