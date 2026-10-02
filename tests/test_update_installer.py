"""Tests for update/installer.py: verifying and starting the downloaded setup program."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from eye_tracker import REPO_URL
from eye_tracker.update import installer
from eye_tracker.update.fetch import Cancelled, Fetcher, NetworkError
from eye_tracker.update.release import Asset, ReleaseInfo, download_url
from test_update_fetch import FakeResponse, FakeTransport

TAG = "v0.2.1"
NAME = "EyeTracker-0.2.1-windows-x64-setup.exe"
BODY = b"MZ" + b"\x00" * 5000


def release(size: int = len(BODY), *, with_checksums: bool = True) -> ReleaseInfo:
    return ReleaseInfo(
        version="0.2.1",
        tag=TAG,
        page_url=f"{REPO_URL}/releases/tag/{TAG}",
        installer=Asset(NAME, download_url(TAG, NAME), size),
        checksums=Asset("SHA256SUMS.txt", download_url(TAG, "SHA256SUMS.txt"), 200)
        if with_checksums
        else None,
    )


def sums(digest: str, name: str = NAME) -> bytes:
    other = "a" * 64
    return f"{other}  EyeTracker-0.2.1-linux-x86_64.tar.gz\n{digest}  {name}\n".encode()


def transport(body: bytes = BODY, listing: bytes | None = None) -> FakeTransport:
    digest = hashlib.sha256(BODY).hexdigest()
    return FakeTransport(
        {
            ("github.com", f"/bugraskl/eye-tracker/releases/download/{TAG}/SHA256SUMS.txt"): (
                FakeResponse(body=listing if listing is not None else sums(digest))
            ),
            ("github.com", f"/bugraskl/eye-tracker/releases/download/{TAG}/{NAME}"): FakeResponse(
                body=body
            ),
        }
    )


# ------------------------------------------------------------------------- checksums
def test_checksum_lists_are_read_like_sha256sum_writes_them() -> None:
    digest = "AB" * 32
    parsed = installer.parse_checksums(
        f"{digest}  a.zip\n{'cd' * 32} *b.exe\nnot a checksum line\n{'e' * 63}  short.zip\n"
    )
    assert parsed == {"a.zip": "ab" * 32, "b.exe": "cd" * 32}


def test_file_sha256(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(BODY)
    assert installer.file_sha256(path) == hashlib.sha256(BODY).hexdigest()


# -------------------------------------------------------------------------- download
def test_a_verified_download_is_kept(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []
    path = installer.download_installer(
        release(),
        Fetcher(transport()),
        progress=lambda d, t: seen.append((d, t)),
        directory=tmp_path,
    )
    assert path == tmp_path / NAME
    assert path.read_bytes() == BODY
    assert sorted(p.name for p in tmp_path.iterdir()) == [NAME]  # no .part left
    assert seen[-1] == (len(BODY), len(BODY))


def test_a_download_with_the_wrong_checksum_is_deleted(tmp_path: Path) -> None:
    tampered = BODY[:-1] + b"X"
    with pytest.raises(installer.UpdateError, match="checksum"):
        installer.download_installer(release(), Fetcher(transport(tampered)), directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_short_download_is_deleted(tmp_path: Path) -> None:
    with pytest.raises(installer.UpdateError, match="incomplete"):
        installer.download_installer(
            release(size=len(BODY) + 10), Fetcher(transport()), directory=tmp_path
        )
    assert list(tmp_path.iterdir()) == []


def test_a_download_larger_than_announced_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NetworkError, match="larger"):
        installer.download_installer(
            release(size=len(BODY) - 1), Fetcher(transport()), directory=tmp_path
        )
    assert list(tmp_path.iterdir()) == []


def test_an_installer_missing_from_the_checksum_list_is_refused(tmp_path: Path) -> None:
    listing = sums(hashlib.sha256(BODY).hexdigest(), name="something-else.exe")
    t = transport(listing=listing)
    with pytest.raises(installer.UpdateError, match="does not name"):
        installer.download_installer(release(), Fetcher(t), directory=tmp_path)
    # The installer itself was never asked for.
    assert all(path.endswith("SHA256SUMS.txt") for _, path, _ in t.asked)


def test_a_release_without_checksums_is_not_installed(tmp_path: Path) -> None:
    with pytest.raises(installer.UpdateError, match="no installer"):
        installer.download_installer(
            release(with_checksums=False), Fetcher(transport()), directory=tmp_path
        )


def test_a_cancelled_download_leaves_nothing(tmp_path: Path) -> None:
    with pytest.raises(Cancelled):
        installer.download_installer(
            release(), Fetcher(transport()), cancelled=lambda: True, directory=tmp_path
        )
    assert list(tmp_path.iterdir()) == []


def test_an_older_download_is_replaced(tmp_path: Path) -> None:
    (tmp_path / NAME).write_bytes(b"old")
    (tmp_path / (NAME + ".part")).write_bytes(b"older")
    path = installer.download_installer(release(), Fetcher(transport()), directory=tmp_path)
    assert path.read_bytes() == BODY
    assert [p.name for p in tmp_path.iterdir()] == [NAME]


def test_leftover_downloads_are_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "updates_dir", lambda: tmp_path)
    (tmp_path / "a.exe").write_bytes(b"1")
    (tmp_path / "b.part").write_bytes(b"2")
    installer.remove_downloads()
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(installer, "updates_dir", lambda: tmp_path / "missing")
    installer.remove_downloads()  # nothing to do, no error


# ---------------------------------------------------------------------------- kinds
def test_a_source_checkout_is_not_updated_by_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.paths, "is_frozen", lambda: False)
    assert installer.install_kind() == installer.KIND_UNSUPPORTED


def test_other_platforms_are_not_updated_by_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.paths, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    assert installer.install_kind() == installer.KIND_UNSUPPORTED


def test_the_installed_app_is_told_by_its_uninstaller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer.paths, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "EyeTracker.exe"))
    assert installer.install_kind() == installer.KIND_PORTABLE
    (tmp_path / "unins000.exe").write_bytes(b"")
    assert installer.install_kind() == installer.KIND_SETUP


# --------------------------------------------------------------------------- launch
def test_the_setup_is_started_silently_and_detached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(installer, "install_kind", lambda: installer.KIND_SETUP)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)) or object())
    path = tmp_path / NAME
    installer.launch_installer(path)
    command, options = calls[0]
    assert command == [str(path), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]
    assert options["close_fds"] is True
    assert options["stdin"] is subprocess.DEVNULL


@pytest.mark.parametrize("kind", [installer.KIND_PORTABLE, installer.KIND_UNSUPPORTED])
def test_the_setup_is_not_started_for_copies_it_does_not_own(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer, "install_kind", lambda: kind)
    started: list[object] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a))
    with pytest.raises(installer.UpdateError):
        installer.launch_installer(tmp_path / NAME)
    assert started == []
