#!/usr/bin/env python3
"""Smoke-test a frozen Eye Tracker build through its command-line interface.

``.github/workflows/build.yml`` runs this for every package it makes, the way users
get it: the Windows bundle (the contents of the portable ZIP) and its installed
copy, the command-line tool inside the mounted macOS disk image, and the Linux
AppImage. Run it from the repository root, in the build environment::

    uv run python scripts/smoke_test_bundle.py --version 0.1.0 dist/EyeTracker/eye-tracker-cli.exe
    uv run python scripts/smoke_test_bundle.py --version 0.1.0 \\
        "/Volumes/Eye Tracker/Eye Tracker.app/Contents/MacOS/eye-tracker-cli"
    uv run python scripts/smoke_test_bundle.py --version 0.1.0 \\
        dist/EyeTracker-0.1.0-linux-x86_64.AppImage

It checks that the build

* reports the expected version and that it is frozen, with every face model present
  and matching its pinned SHA-256 (``--version``, ``doctor --json``);
* analyses an image with every backend (``bench --camera <png> --backend ...``);
* decodes video files: an AVI written by OpenCV's own MJPEG encoder and an MP4
  (MPEG-4 Part 2) written by FFmpeg, both with this environment's OpenCV, which the
  build reads with the libraries it bundles;
* probes the cameras with each platform's capture backend (``doctor
  --probe-cameras``). The CI runners have no camera; on macOS, OpenCV's message
  for a camera it could not open shows that AVFoundation was asked.

The tool writes its settings and logs into a temporary folder, never into the
user's. Exit status: 0 when every check passed, 1 otherwise (the failed check and
the tool's output are printed), 2 on a usage error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
#: A picture to analyse (the app's icon: no face, but every frame gets analysed).
IMAGE = REPO_ROOT / "packaging" / "linux" / "eye-tracker.png"
#: (file name, OpenCV writer back end, FourCC) of the video files to decode.
CLIPS = (
    ("clip.avi", "CAP_OPENCV_MJPEG", "MJPG"),
    ("clip.mp4", "CAP_FFMPEG", "mp4v"),
)
CLIP_SIZE = (160, 120)
#: Printed by OpenCV's AVFoundation backend for each camera it cannot open.
AVFOUNDATION_PROBE_MESSAGE = "OpenCV: camera failed to properly initialize!"
#: Seconds one command may take (the AppImage unpacks itself on every start).
TIMEOUT = 300


class SmokeTestError(Exception):
    """A check failed."""


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``command``; a non-zero exit status is a failure that shows the output."""
    print("$ " + subprocess.list2cmdline(command), flush=True)
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmokeTestError(f"could not run {command[0]}: {exc}") from exc
    if proc.returncode != 0:
        raise SmokeTestError(
            f"exit status {proc.returncode}\n--- stdout ---\n{proc.stdout}"
            f"\n--- stderr ---\n{proc.stderr}"
        )
    return proc


def _json(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SmokeTestError(
            f"the output is not JSON ({exc})\n--- stdout ---\n{proc.stdout}"
            f"\n--- stderr ---\n{proc.stderr}"
        ) from exc
    if not isinstance(data, dict):
        raise SmokeTestError(f"expected a JSON object, got: {proc.stdout}")
    return data


def _check(condition: bool, message: str, details: object = None) -> None:
    if not condition:
        raise SmokeTestError(message if details is None else f"{message}: {details!r}")


def write_clips(folder: Path) -> list[Path]:
    """Ten frames of growing brightness in every format of :data:`CLIPS`."""
    import cv2
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    width, height = CLIP_SIZE
    written = []
    for name, api, codec in CLIPS:
        path = folder / name
        fourcc = cv2.VideoWriter.fourcc(*codec)
        writer = cv2.VideoWriter(str(path), getattr(cv2, api), fourcc, 10.0, (width, height))
        _check(writer.isOpened(), f"this environment's OpenCV cannot write {name}")
        for level in range(0, 250, 25):
            writer.write(np.full((height, width, 3), level, np.uint8))
        writer.release()
        written.append(path)
    return written


def smoke_test(cli: Path, version: str, work: Path) -> None:
    """Run every check against the command-line tool ``cli``."""
    from eye_tracker.vision.backends import BACKEND_NAMES, MODEL_FILES

    tool = [str(cli)]
    config = [*tool, "--config-dir", str(work / "config")]

    output = _run([*tool, "--version"]).stdout.strip()
    _check(version in output.split(), f"--version does not report {version}", output)
    print(f"ok: {output}")

    report = _json(_run([*config, "doctor", "--json"]))
    app = report["app"]
    _check(app["version"] == version, f"doctor reports another version than {version}", app)
    _check(app["frozen"] is True, "doctor does not run frozen", app)
    models = report["backends"]["models"]
    _check(set(models) == set(MODEL_FILES), "doctor lists other face models", sorted(models))
    for name, model in models.items():
        _check(bool(model.get("present") and model.get("sha256_ok")), f"model {name}", model)
    print(f"ok: doctor (frozen, {len(models)} face models present and verified)")
    print(_run([*config, "doctor"]).stdout)

    for backend in BACKEND_NAMES:
        image = ["--camera", str(IMAGE), "--backend", backend]
        bench = _json(_run([*config, "bench", *image, "--seconds", "1", "--json"]))
        _check(bench["backend"] == backend, f"bench ran another backend than {backend}", bench)
        analysed = bench["modes"]["max"]["analysed"]
        _check(analysed > 0, f"the {backend} backend analysed no frame", bench["modes"])
        print(f"ok: the {backend} backend analysed {analysed} frames of {IMAGE.name}")

    for clip in write_clips(work / "clips"):
        video = _json(_run([*config, "bench", "--camera", str(clip), "--seconds", "1", "--json"]))
        _check(video["device"] == f"file {clip.name}", "bench read another source", video)
        _check(video["frame_size"] == list(CLIP_SIZE), f"{clip.name} decoded wrongly", video)
        analysed = video["modes"]["max"]["analysed"]
        _check(analysed > 0, f"no frame of {clip.name} was analysed", video["modes"])
        print(f"ok: decoded {clip.name} and analysed {analysed} frames")

    probe = _run([*config, "doctor", "--probe-cameras", "--json"])
    cameras = _json(probe)["cameras"]
    _check(cameras["probed"] is True, "the cameras were not probed", cameras)
    _check(isinstance(cameras["devices"], list), "no list of cameras", cameras)
    if sys.platform == "darwin" and not cameras["devices"]:
        _check(
            AVFOUNDATION_PROBE_MESSAGE in probe.stderr,
            "probing did not reach AVFoundation",
            probe.stderr,
        )
    print(f"ok: probed the cameras ({len(cameras['devices'])} found)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test a frozen Eye Tracker build through its command-line interface."
    )
    parser.add_argument("--version", required=True, help="the version the build must report")
    parser.add_argument("cli", type=Path, help="the frozen command-line tool (or the AppImage)")
    args = parser.parse_args(argv)
    cli = args.cli.resolve()
    if not cli.is_file():
        parser.error(f"{cli} does not exist")
    # The frozen tool may still hold a log file open for a moment on Windows.
    with tempfile.TemporaryDirectory(
        prefix="eye-tracker-smoke-", ignore_cleanup_errors=True
    ) as work:
        try:
            smoke_test(cli, args.version, Path(work))
        except SmokeTestError as exc:
            print(f"smoke test failed: {exc}", file=sys.stderr)
            return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
