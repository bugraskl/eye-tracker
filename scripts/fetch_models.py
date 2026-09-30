#!/usr/bin/env python3
"""Download and verify the face models that ship with Eye Tracker.

The models are committed to the repository, so most people never need to run
this. It exists to recreate them in a fresh or damaged checkout, to verify them
in CI, and to record exactly where each file comes from and under which licence.

Usage::

    python scripts/fetch_models.py            # download missing or corrupt models
    python scripts/fetch_models.py --check    # verify only; never touches the network
    python scripts/fetch_models.py --force    # download again even if valid

Every file is checked against a pinned SHA-256 before it is moved into place,
so a failed or tampered download can never replace a good model.

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
class Model:
    """A model file with its canonical source and pinned checksum."""

    filename: str
    url: str
    sha256: str
    license: str
    homepage: str
    description: str


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
        description="MediaPipe Face Landmarker (478 landmarks incl. iris, head pose)",
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
        description="OpenCV Zoo YuNet face detector (lightweight fallback backend)",
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
    """``"ok"``, ``"missing"`` or ``"mismatch"`` for the model file in ``dest``."""
    path = dest / model.filename
    if not path.is_file():
        return "missing"
    return "ok" if sha256_file(path) == model.sha256 else "mismatch"


def _human_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB"):
        if value < 1024 or unit == "MiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"  # pragma: no cover - loop always returns


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


def _pause_before_retry(model: Model, exc: Exception, attempt: int) -> None:
    delay = 2.0 * attempt
    print(f"  retry    {model.filename}: {exc} (again in {delay:.0f} s)")
    time.sleep(delay)


def download(model: Model, dest: Path, *, timeout: float = 60.0, attempts: int = 3) -> Path:
    """Download ``model`` into ``dest`` atomically; returns the final path.

    The data goes to a temporary file in the same directory and replaces the
    target only after the checksum matches.
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
            size = _human_size((dest / model.filename).stat().st_size)
            print(f"  ok       {model.filename} ({size}, sha256 {model.sha256[:12]}...)")
            continue
        if args.check:
            print(f"  {status:<8} {model.filename}  (run scripts/fetch_models.py to fix)")
            failures += 1
            continue
        print(f"  fetch    {model.filename} [{model.license}] from {model.url}")
        try:
            path = download(model, dest, timeout=args.timeout)
        except DownloadError as exc:
            print(f"  FAILED   {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"  ok       {model.filename} ({_human_size(path.stat().st_size)}, verified)")

    if failures:
        print(f"{failures} model(s) missing or invalid in {dest}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
