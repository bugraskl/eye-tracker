#!/usr/bin/env python3
"""Download and verify the face models that ship with Eye Tracker.

The models are committed to the repository, so most people never need to run
this. It exists to recreate them in a fresh or damaged checkout, to verify them
in CI, and to record exactly where each file comes from and under which licence
(see also ``src/eye_tracker/vision/models/NOTICE.md``).

Usage::

    python scripts/fetch_models.py            # download missing or corrupt models
    python scripts/fetch_models.py --check    # verify only; never touches the network
    python scripts/fetch_models.py --force    # download again even if valid

Every download is checked against a pinned SHA-256 before anything is moved
into place, so a failed or tampered download can never replace a good model.
Some models come inside an archive (MediaPipe's ``face_landmarker.task`` is a
zip file): the archive is verified, only the members the app needs are
extracted, and each member is verified against its own pinned SHA-256. The
archive itself is not kept.

The models' licence texts (``licenses/`` next to the models) must ship with
them. They are committed like the models but never downloaded: ``--check``
fails when one is missing or altered, the other modes only warn.

This is developer tooling that lives outside ``src/``: the application itself
never touches the network.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "src" / "eye_tracker" / "vision" / "models"
USER_AGENT = "eye-tracker-fetch-models/1.0 (+https://github.com/bugraskl/eye-tracker)"

#: Refuse absurdly large responses (the real files are a few MiB).
MAX_BYTES = 64 * 1024 * 1024
_CHUNK = 1024 * 256


@dataclass(frozen=True)
class Member:
    """A file installed from a downloaded archive."""

    #: Path inside the archive; also the installed file name.
    name: str
    sha256: str


@dataclass(frozen=True)
class Model:
    """A model download with its canonical source and pinned checksum.

    With ``members`` the download is a zip archive and only those members are
    installed; otherwise the downloaded file is installed as ``filename``.
    """

    filename: str
    url: str
    sha256: str
    license: str
    homepage: str
    description: str
    members: tuple[Member, ...] = ()

    def installed(self) -> tuple[tuple[str, str], ...]:
        """``(file name, sha256)`` of every file this model puts into the model directory."""
        if self.members:
            return tuple((m.name, m.sha256) for m in self.members)
        return ((self.filename, self.sha256),)


MODELS: tuple[Model, ...] = (
    Model(
        filename="face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        license="Apache-2.0",
        homepage="https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
        description=(
            "MediaPipe Face Landmarker bundle: the 478-point face landmark network "
            "(with irises) and the canonical face geometry, run with OpenCV DNN"
        ),
        members=(
            Member(
                "face_landmarks_detector.tflite",
                "c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5",
            ),
            Member(
                "geometry_pipeline_metadata_landmarks.binarypb",
                "bdbcda96dfcb7da883da124aaa2c55dee49770d934f0fcc71747f8c21bdc75b4",
            ),
        ),
    ),
    Model(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        license="MIT",
        homepage="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        description="OpenCV Zoo YuNet face detector (face finding and the lite backend)",
    ),
)


@dataclass(frozen=True)
class LicenceText:
    """A licence text that ships next to the models it covers (see NOTICE.md).

    Apache-2.0 requires giving recipients a copy of the licence, and MIT
    requires its copyright and permission notice in all copies, so a link is not
    enough. The texts are committed to the repository and only verified.
    """

    #: Path relative to the model directory.
    path: str
    sha256: str
    #: Where the text comes from.
    source: str
    #: Installed model files the licence covers.
    covers: tuple[str, ...]


LICENCE_TEXTS: tuple[LicenceText, ...] = (
    LicenceText(
        path="licenses/LICENSE-APACHE-2.0.txt",
        sha256="cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        source="https://www.apache.org/licenses/LICENSE-2.0.txt",
        covers=("face_landmarks_detector.tflite", "geometry_pipeline_metadata_landmarks.binarypb"),
    ),
    LicenceText(
        path="licenses/LICENSE-YUNET.txt",
        sha256="2ad92c7a6eb7aebede4e19f5ec4930c8bd9d614dbb8e75be1f740edae346734c",
        source=(
            "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE"
        ),
        covers=("face_detection_yunet_2023mar.onnx",),
    ),
)


class DownloadError(RuntimeError):
    """A model could not be downloaded or failed verification."""


# ------------------------------------------------------------------------------ helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model: Model, dest: Path) -> str:
    """``"ok"``, ``"missing"`` or ``"mismatch"`` for the model's files in ``dest``."""
    status = "ok"
    for name, sha256 in model.installed():
        path = dest / name
        if not path.is_file():
            return "missing"
        if sha256_file(path) != sha256:
            status = "mismatch"
    return status


def verify_licence(text: LicenceText, dest: Path) -> str:
    """``"ok"``, ``"missing"`` or ``"altered"`` for a licence text in ``dest``."""
    path = dest / text.path
    if not path.is_file():
        return "missing"
    return "ok" if sha256_file(path) == text.sha256 else "altered"


def check_licences(dest: Path) -> int:
    """Report every licence text's status; returns how many are missing or altered."""
    problems = 0
    for text in LICENCE_TEXTS:
        status = verify_licence(text, dest)
        if status == "ok":
            print(f"  licence  {text.path} (sha256 verified)")
            continue
        problems += 1
        print(f"  {status:<8} {text.path}  (restore it from git: it must ship with the models)")
    return problems


def _human_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB"):
        if value < 1024 or unit == "MiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"  # pragma: no cover - loop always returns


def _installed_size(model: Model, dest: Path) -> int:
    return sum((dest / name).stat().st_size for name, _ in model.installed())


def _download_once(model: Model, target: Path, timeout: float) -> None:
    request = urllib.request.Request(model.url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    received = 0
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as out:
        while chunk := response.read(_CHUNK):
            received += len(chunk)
            if received > MAX_BYTES:
                raise DownloadError(f"{model.filename}: response larger than {MAX_BYTES} bytes")
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != model.sha256:
        raise DownloadError(
            f"{model.filename}: checksum mismatch (got {digest.hexdigest()}, "
            f"expected {model.sha256}); the upstream file may have changed"
        )


def _write_atomic(data: bytes, final: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".part", dir=final.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)


def extract_members(model: Model, archive: Path, dest: Path) -> list[Path]:
    """Install the verified members of a downloaded (and verified) zip archive.

    Every member is read and checked before any file is written, so a bad
    archive leaves the model directory untouched.

    Raises:
        DownloadError: The archive is not a zip file, lacks a member or a
            member's checksum does not match.
    """
    contents: list[tuple[Member, bytes]] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in model.members:
                try:
                    data = bundle.read(member.name)
                except KeyError as exc:
                    raise DownloadError(
                        f"{model.filename}: archive has no member {member.name!r}"
                    ) from exc
                actual = hashlib.sha256(data).hexdigest()
                if actual != member.sha256:
                    raise DownloadError(
                        f"{model.filename}: checksum mismatch for {member.name} "
                        f"(got {actual}, expected {member.sha256})"
                    )
                contents.append((member, data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise DownloadError(f"{model.filename}: not a readable zip archive: {exc}") from exc
    installed = []
    for member, data in contents:
        # Only the pinned base name is used: archive paths never reach the disk.
        final = dest / Path(member.name).name
        _write_atomic(data, final)
        installed.append(final)
    return installed


def _pause_before_retry(model: Model, exc: Exception, attempt: int) -> None:
    delay = 2.0 * attempt
    print(f"  retry    {model.filename}: {exc} (again in {delay:.0f} s)")
    time.sleep(delay)


def download(model: Model, dest: Path, *, timeout: float = 60.0, attempts: int = 3) -> Path:
    """Download ``model`` into ``dest`` atomically; returns the (first) installed path.

    The data goes to a temporary file in the same directory and is installed
    only after the checksum matches (for archives: after every member matched).
    """
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / model.filename
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        fd, tmp_name = tempfile.mkstemp(prefix=f".{model.filename}.", suffix=".part", dir=dest)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            _download_once(model, tmp, timeout)
            if model.members:
                return extract_members(model, tmp, dest)[0]
            os.replace(tmp, final)
            return final
        except DownloadError:
            raise  # a checksum mismatch will not fix itself on retry
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                raise DownloadError(f"{model.filename}: {model.url} returned {exc}") from exc
            last_error = exc
            if attempt < attempts:
                _pause_before_retry(model, exc, attempt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                _pause_before_retry(model, exc, attempt)
        finally:
            tmp.unlink(missing_ok=True)
    raise DownloadError(f"{model.filename}: download failed: {last_error}")


def _describe(model: Model) -> str:
    if not model.members:
        return model.filename
    return f"{model.filename} -> " + ", ".join(m.name for m in model.members)


# --------------------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify Eye Tracker's face models.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="verify only; exit 1 if anything is wrong"
    )
    mode.add_argument("--force", action="store_true", help="download even if the files are valid")
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST, help="model directory (default: %(default)s)"
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="network timeout in seconds (default: 60)"
    )
    args = parser.parse_args(argv)
    dest: Path = args.dest

    failures = 0
    for model in MODELS:
        status = "stale" if args.force else verify(model, dest)
        if status == "ok":
            size = _human_size(_installed_size(model, dest))
            print(f"  ok       {_describe(model)} ({size}, sha256 verified)")
            continue
        if args.check:
            print(f"  {status:<8} {_describe(model)}  (run scripts/fetch_models.py to fix)")
            failures += 1
            continue
        print(f"  fetch    {model.filename} [{model.license}] from {model.url}")
        try:
            download(model, dest, timeout=args.timeout)
        except DownloadError as exc:
            print(f"  FAILED   {exc}", file=sys.stderr)
            failures += 1
            continue
        size = _human_size(_installed_size(model, dest))
        print(f"  ok       {_describe(model)} ({size}, verified)")

    licence_problems = check_licences(dest)
    if failures:
        print(f"{failures} model(s) missing or invalid in {dest}", file=sys.stderr)
    if licence_problems:
        # Committed files, never downloaded: only --check (CI, release builds)
        # fails on them; fetching models into another folder merely warns.
        prefix = "" if args.check else "warning: "
        print(
            f"{prefix}{licence_problems} licence text(s) missing or altered in {dest}",
            file=sys.stderr,
        )
    return 1 if failures or (args.check and licence_problems) else 0


if __name__ == "__main__":
    sys.exit(main())
