"""Frame sources: webcams through OpenCV and video/image files for tests and demos.

Every source hands out BGR ``uint8`` frames that live only in memory. Nothing in
this module writes image data anywhere.

Sources are used from the vision worker thread. ``open()`` may block for a few
seconds (some Windows camera drivers are slow to start) but never raises;
failures are reported through ``last_error``.

Offline guarantee: :func:`open_source` and :class:`FileSource` refuse URLs and
other protocol specifications, so only local files are ever opened, and video
files are read by OpenCV's FFmpeg backend alone. FFmpeg only lets a local file
reference local data (its protocol whitelist for ``file`` inputs is
``file,crypto,data``), so a playlist or SDP file cannot pull a network stream
either; other backends (GStreamer in distribution builds of OpenCV, Media
Foundation) are never asked, because they would follow such references.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import sys
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: File suffixes treated as still images by :class:`FileSource`.
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".pgm", ".ppm", ".pbm"}
)

# A camera that fails this many reads in a row, or keeps failing for this long,
# is considered gone (unplugged, taken over by another application or its
# driver crashed). The time bound matters for drivers whose reads block for
# seconds before failing (Media Foundation, V4L2 select timeouts).
_MAX_READ_FAILURES = 5
_MAX_FAILURE_S = 3.0
# Reads attempted while opening; the first frames of some cameras are empty.
_WARMUP_READS = 5
# YUY2 bandwidth (bytes/s) above which DirectShow is asked for MJPG: USB 2
# cameras cannot deliver more uncompressed, while below it the uncompressed
# format is cheaper to handle than DirectShow's MJPEG decoder.
_DSHOW_MJPG_ABOVE_BYTES_S = 20e6

_API_BY_NAME: dict[str, int] = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "v4l2": cv2.CAP_V4L2,
}

_DEV_VIDEO = re.compile(r"^/dev/video(\d+)$")
# What FFmpeg would open as a protocol rather than a file (see its
# url_find_protocol): "<scheme>:…" where the scheme is letters, digits, "+", "-"
# or "." ("rtsp://…", "http:…", "tcp:…", "concat:…"), except a single drive
# letter ("C:\…"); "subfile,…:" wraps another protocol; and "://" anywhere
# catches nested URLs.
_URL_LIKE = re.compile(r"^(?:[A-Za-z0-9+.\-]{2,}:|subfile,)|://", re.IGNORECASE)

if sys.platform.startswith("linux"):
    # V4L2 waits up to 10 s (the default) for each frame of a stalled camera,
    # during which the worker cannot release it. OpenCV reads this on the first
    # capture, so setting it at import time is early enough.
    os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "2")


class CameraError(RuntimeError):
    """A frame source could not be created (for example a missing video file)."""


@dataclass
class CameraInfo:
    """A camera found by :func:`list_cameras`."""

    index: int
    name: str
    width: int
    height: int


@runtime_checkable
class FrameSource(Protocol):
    """Anything that produces BGR frames for the vision worker."""

    def open(self) -> bool:
        """Start capturing. Returns ``False`` (and sets ``last_error``) on failure."""
        ...

    def read(self) -> np.ndarray | None:
        """Return the most recent BGR frame, or ``None`` if none is available."""
        ...

    def release(self) -> None:
        """Stop capturing and free the device. Safe to call more than once."""
        ...

    @property
    def is_open(self) -> bool: ...

    @property
    def last_error(self) -> str | None: ...


def api_preference(api: str) -> int:
    """Map a capture API name from the settings to an OpenCV ``CAP_*`` constant.

    ``"auto"`` picks the API that opens fastest and supports MJPG on each OS:
    DirectShow on Windows (Media Foundation can take many seconds to open),
    AVFoundation on macOS and V4L2 on Linux.
    """
    key = api.strip().lower()
    if key == "auto":
        if sys.platform == "win32":
            return cv2.CAP_DSHOW
        if sys.platform == "darwin":
            return cv2.CAP_AVFOUNDATION
        if sys.platform.startswith("linux"):
            return cv2.CAP_V4L2
        return cv2.CAP_ANY
    try:
        return _API_BY_NAME[key]
    except KeyError:
        choices = ", ".join(["auto", *_API_BY_NAME])
        raise ValueError(f"Unknown camera API {api!r}; expected one of {choices}") from None


def _fourcc(code: str) -> int:
    # Computed by hand: the equivalent OpenCV helper lives on the video writer
    # class, which this package deliberately never references.
    a, b, c, d = (ord(ch) for ch in code)
    return a | (b << 8) | (c << 16) | (d << 24)


def _fourcc_name(cap: cv2.VideoCapture) -> str:
    """The pixel format the driver reports, e.g. ``"MJPG"`` or ``"YUY2"`` (diagnostics)."""
    try:
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
    except (cv2.error, ValueError, OverflowError):
        return "?"
    text = "".join(chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24))
    return text if text.isprintable() and text.strip() else f"0x{code:08x}"


def _backend_name(cap: cv2.VideoCapture) -> str:
    try:
        return str(cap.getBackendName())
    except cv2.error:
        return "?"


def _try_set(cap: cv2.VideoCapture, prop: int, value: float) -> None:
    # Drivers silently ignore properties they do not support; some raise instead.
    with contextlib.suppress(cv2.error):
        cap.set(prop, value)


def _valid_frame(frame: np.ndarray | None) -> bool:
    return frame is not None and frame.size > 0


class Camera(FrameSource):
    """A webcam opened through ``cv2.VideoCapture``.

    Args:
        index: Device index (``0`` is the default camera).
        width, height: Requested capture size. Small sizes keep USB bandwidth and
            decoding cost low; the backends downscale further anyway.
        api: Capture API name, see :func:`api_preference`.
        fps: Requested camera frame rate. Some drivers convert every captured
            frame, so asking for fewer frames saves CPU even between reads.
        flush: Buffered frames to drop before a read that follows a pause, so
            that a slow reader never analyses a stale picture.
    """

    def __init__(
        self,
        index: int,
        width: int = 640,
        height: int = 480,
        api: str = "auto",
        fps: int = 15,
        flush: int = 1,
    ) -> None:
        if index < 0:
            raise ValueError(f"camera index must be >= 0, got {index}")
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.api = api
        self.fps = max(1, int(fps))
        self.flush = max(0, int(flush))
        #: The settings string this camera was configured with (``"0"``,
        #: ``"/dev/v4l/by-id/…"``); identifies the camera for calibrations.
        self.device = str(self.index)
        self._api_pref = api_preference(api)
        self._cap: cv2.VideoCapture | None = None
        self._error: str | None = None
        self._failures = 0
        self._fail_since: float | None = None
        self._last_read = -math.inf
        self._frame_size: tuple[int, int] | None = None

    # ------------------------------------------------------------------ info
    @property
    def name(self) -> str:
        return f"Camera {self.index}"

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def frame_size(self) -> tuple[int, int] | None:
        """``(width, height)`` of the delivered frames once the camera is open."""
        return self._frame_size

    # ------------------------------------------------------------- lifecycle
    def open(self) -> bool:
        if self._cap is not None:
            return True
        self._failures = 0
        self._fail_since = None
        started = time.monotonic()
        # The OS privacy switches (Windows "Let desktop apps access your camera",
        # macOS camera access) make opening fail exactly like a busy camera.
        unavailable = (
            f"{self.name} could not be opened — it may be in use by another app "
            "or blocked by the system's camera privacy settings"
        )
        try:
            cap = cv2.VideoCapture(self.index, self._api_pref)
        except Exception as exc:  # open() must never raise; drivers fail in odd ways
            log.debug("VideoCapture(%d) raised: %s", self.index, exc)
            self._error = unavailable
            return False
        if not cap.isOpened():
            cap.release()
            self._error = unavailable
            log.debug("%s did not open (%.0f ms)", self.name, (time.monotonic() - started) * 1e3)
            return False

        if self._api_pref == cv2.CAP_DSHOW:
            self._configure_dshow(cap)
        else:
            # FOURCC first: on V4L2 changing the pixel format afterwards can reset
            # the frame size. MJPG lets USB 2 cameras deliver 640x480 at full
            # rate; cameras without MJPG simply keep their default format.
            _try_set(cap, cv2.CAP_PROP_FOURCC, _fourcc("MJPG"))
            _try_set(cap, cv2.CAP_PROP_FRAME_WIDTH, self.width)
            _try_set(cap, cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            _try_set(cap, cv2.CAP_PROP_FPS, self.fps)
        _try_set(cap, cv2.CAP_PROP_BUFFERSIZE, 1)

        # Some drivers report success on open but never deliver a frame when another
        # application holds the device, so only a real frame counts as success.
        frame: np.ndarray | None = None
        for _ in range(_WARMUP_READS):
            try:
                ok, frame = cap.read()
            except Exception as exc:
                log.debug("%s warm-up read raised: %s", self.name, exc)
                ok, frame = False, None
            if ok and _valid_frame(frame):
                break
            frame = None
        if frame is None:
            cap.release()
            self._error = (
                f"{self.name} opened but delivers no frames — it may be in use by another "
                "app or blocked by the system's camera privacy settings"
            )
            log.debug("%s delivered no frames", self.name)
            return False

        self._cap = cap
        self._error = None
        self._last_read = time.monotonic()
        self._frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "%s opened via %s at %dx%d (%s) in %.0f ms",
                self.name,
                _backend_name(cap),
                self._frame_size[0],
                self._frame_size[1],
                _fourcc_name(cap),
                (time.monotonic() - started) * 1e3,
            )
        return True

    def _configure_dshow(self, cap: cv2.VideoCapture) -> None:
        """Apply size, rate and format with as few DirectShow graph rebuilds as possible.

        OpenCV's DirectShow backend rebuilds the capture graph for every format
        change, and a size or rate change renegotiates the pixel format from
        scratch (dropping a previously chosen MJPG). So only values that differ
        are set, size before rate, and the MJPG request comes last (DirectShow
        keeps the current size and rate when changing the format) and only when
        uncompressed frames would exceed USB 2 bandwidth.
        """

        def current(prop: int) -> float:
            try:
                return float(cap.get(prop))
            except cv2.error:
                return 0.0

        size = (round(current(cv2.CAP_PROP_FRAME_WIDTH)), round(current(cv2.CAP_PROP_FRAME_HEIGHT)))
        if size != (self.width, self.height):
            _try_set(cap, cv2.CAP_PROP_FRAME_WIDTH, self.width)
            _try_set(cap, cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if round(current(cv2.CAP_PROP_FPS)) != self.fps:
            _try_set(cap, cv2.CAP_PROP_FPS, self.fps)
        if self.width * self.height * 2 * self.fps > _DSHOW_MJPG_ABOVE_BYTES_S:
            _try_set(cap, cv2.CAP_PROP_FOURCC, _fourcc("MJPG"))

    def read(self) -> np.ndarray | None:
        cap = self._cap
        if cap is None:
            return None
        frame: np.ndarray | None = None
        ok = False
        try:
            # Drop buffered frames only after a pause longer than ~1.5 frame periods.
            # When reading at camera speed nothing stale can be queued, and grabbing
            # anyway would block for an extra frame period and halve the frame rate.
            # Never while failing: a stalled driver would block once per grab.
            flushed = True
            paused = time.monotonic() - self._last_read > 1.5 / self.fps
            if self.flush and paused and self._failures == 0:
                for _ in range(self.flush):
                    if not cap.grab():
                        flushed = False
                        break
            if flushed:
                ok, frame = cap.read()
        except Exception as exc:
            log.debug("%s read raised: %s", self.name, exc)
        now = time.monotonic()
        self._last_read = now

        if not ok or not _valid_frame(frame):
            self._failures += 1
            if self._fail_since is None:
                self._fail_since = now
            if self._failures >= _MAX_READ_FAILURES or now - self._fail_since >= _MAX_FAILURE_S:
                self._error = (
                    f"{self.name} stopped delivering frames — was it unplugged or taken "
                    "by another app?"
                )
                log.debug("%s stopped delivering frames; closing it", self.name)
                self.release()
            return None
        self._failures = 0
        self._fail_since = None
        return frame

    def release(self) -> None:
        cap, self._cap = self._cap, None
        self._failures = 0
        self._fail_since = None
        if cap is not None:
            try:
                cap.release()
            except cv2.error as exc:
                log.debug("%s release raised: %s", self.name, exc)
            log.debug("%s released", self.name)

    def __repr__(self) -> str:
        return f"Camera(index={self.index}, api={self.api!r}, open={self.is_open})"


def _imread(path: Path) -> np.ndarray | None:
    # cv2.imread cannot open non-ASCII paths on Windows; decoding bytes read by
    # Python works everywhere.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        log.debug("Could not read %s: %s", path, exc)
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class FileSource(FrameSource):
    """Frames from a video file (looping forever) or a still image.

    Intended for tests, demos and benchmarks without a camera.

    Args:
        path: Video or image file.
        width, height: Optional bounding box. Larger frames are scaled down to fit
            inside it (aspect ratio preserved, never enlarged), mimicking the
            capture size a camera would deliver.
        realtime: For videos, advance through the file at its native frame rate
            (like a live camera: slow readers skip frames). When ``False`` every
            ``read()`` returns the next frame.
    """

    def __init__(
        self,
        path: str | Path,
        width: int | None = None,
        height: int | None = None,
        *,
        realtime: bool = False,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.realtime = realtime
        #: The settings string this source was configured with.
        self.device = str(self.path)
        self._image: np.ndarray | None = None
        self._cap: cv2.VideoCapture | None = None
        self._error: str | None = None
        self._video_fps = 30.0
        self._last_frame: np.ndarray | None = None
        self._t_prev: float | None = None
        self._debt = 0.0
        # Replaceable for deterministic tests of realtime playback.
        self._now: Callable[[], float] = time.monotonic

    @property
    def is_open(self) -> bool:
        return self._image is not None or self._cap is not None

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def is_image(self) -> bool:
        return self.path.suffix.lower() in IMAGE_SUFFIXES

    def open(self) -> bool:
        if self.is_open:
            return True
        if _URL_LIKE.search(str(self.path)):
            # FFmpeg would treat "rtsp:…", "tcp:…" and the like as network URLs.
            self._error = f"Not a local file: {self.path} (network sources are not supported)"
            return False
        if not self.path.is_file():
            self._error = f"Video source not found: {self.path}"
            return False
        if self.is_image:
            image = _imread(self.path)
            if image is None:
                self._error = f"Could not decode image {self.path.name}"
                return False
            self._image = self._fit(image)
            self._error = None
            return True

        cap = self._open_capture()
        if cap is None:
            self._error = f"Could not open video {self.path.name}"
            return False
        ok, frame = cap.read()
        cap.release()
        if not ok or not _valid_frame(frame):
            self._error = f"Video {self.path.name} contains no readable frames"
            return False
        # Re-open instead of seeking back: seeking is unreliable for some codecs.
        self._cap = self._open_capture()
        if self._cap is None:
            self._error = f"Could not open video {self.path.name}"
            return False
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._video_fps = fps if math.isfinite(fps) and 1.0 <= fps <= 240.0 else 30.0
        self._last_frame = None
        self._t_prev = None
        self._debt = 0.0
        self._error = None
        return True

    def read(self) -> np.ndarray | None:
        if self._image is not None:
            # A copy, so a consumer drawing on the frame cannot alter later reads.
            return self._image.copy()
        if self._cap is None:
            return None
        frame = self._read_realtime() if self.realtime else self._step(decode=True)
        if frame is None:
            return None
        return self._fit(frame)

    def release(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
        self._image = None
        self._last_frame = None

    # --------------------------------------------------------------- helpers
    def _open_capture(self) -> cv2.VideoCapture | None:
        try:
            # FFmpeg only: its protocol whitelist keeps a local file from
            # referencing network data (see the module docs).
            cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        except cv2.error as exc:
            log.debug("VideoCapture(%s) raised: %s", self.path.name, exc)
            return None
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _step(self, *, decode: bool) -> np.ndarray | None:
        """Advance one frame, looping at the end. Returns the frame if decoded."""
        for attempt in range(2):
            cap = self._cap
            if cap is None:
                return None
            if decode:
                ok, frame = cap.read()
                if ok and _valid_frame(frame):
                    self._last_frame = frame
                    return frame
            elif cap.grab():
                return None
            if attempt == 0:
                # End of the file: start over.
                cap.release()
                self._cap = self._open_capture()
        self._error = f"Video {self.path.name} stopped delivering frames"
        self.release()
        return None

    def _read_realtime(self) -> np.ndarray | None:
        now = self._now()
        if self._t_prev is None or self._last_frame is None:
            self._t_prev = now
            self._debt = 0.0
            return self._step(decode=True)
        self._debt += (now - self._t_prev) * self._video_fps
        self._t_prev = now
        steps = int(self._debt)
        self._debt -= steps
        if steps <= 0:
            return self._last_frame.copy()
        # After a long pause jump ahead a little instead of decoding a backlog.
        steps = min(steps, 30)
        for _ in range(steps - 1):
            self._step(decode=False)
            if self._cap is None:
                return None
        return self._step(decode=True)

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = 1.0
        if self.width:
            scale = min(scale, self.width / w)
        if self.height:
            scale = min(scale, self.height / h)
        if scale >= 1.0:
            return frame
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    def __repr__(self) -> str:
        return f"FileSource({self.path.name!r}, open={self.is_open})"


def open_source(
    device: str,
    width: int,
    height: int,
    api: str,
    *,
    fps: int = 15,
    realtime: bool = True,
) -> FrameSource:
    """Create (but do not open) the frame source described by a settings string.

    ``"0"``, ``"1"``, … select a camera by index. On Linux ``/dev/videoN`` and
    stable links to it (``/dev/v4l/by-id/…``, ``/dev/v4l/by-path/…``) are
    accepted too; links are resolved on every call, so a camera that was
    re-plugged under another number is still found. Anything else is a path to
    a local video or image file. URLs and other protocols are refused: Eye
    Tracker never opens network streams.

    Raises:
        CameraError: The file does not exist, is a URL, or the settings are invalid.
    """
    spec = device.strip()
    if not spec:
        raise CameraError("No camera configured")
    if _URL_LIKE.search(spec):
        raise CameraError(
            f"Network or protocol video sources are not supported ({spec!r}); "
            "use a camera index or a local file"
        )
    index: int | None = None
    if spec.isdigit():
        index = int(spec)
    elif sys.platform.startswith("linux") and spec.startswith("/dev/") and not os.path.isfile(spec):
        index = _v4l2_index(spec)
    if index is not None:
        try:
            source = Camera(index, width, height, api, fps=fps)
        except ValueError as exc:
            raise CameraError(str(exc)) from exc
        source.device = spec
        return source

    path = Path(spec).expanduser()
    if not path.is_file():
        raise CameraError(f"Video source not found: {path}")
    file_source = FileSource(path, width, height, realtime=realtime)
    file_source.device = spec
    return file_source


def device_index(device: str) -> int | None:
    """The camera index a ``camera.device`` setting selects right now, or ``None``.

    ``"0"``, ``"1"``, … and, on Linux, ``/dev/videoN`` or a stable link to one
    (``/dev/v4l/by-id/…``, resolved now). ``None`` for a video or image file and
    for a device path that does not lead to a camera. Never raises.
    """
    spec = device.strip()
    if spec.isdigit():
        return int(spec)
    if sys.platform.startswith("linux") and spec.startswith("/dev/") and not os.path.isfile(spec):
        try:
            return _v4l2_index(spec)
        except (CameraError, OSError, ValueError):
            return None
    return None


def _resolve_link(path: str) -> str:
    """``os.path.realpath`` (replaceable in tests)."""
    return os.path.realpath(path)


def _v4l2_index(spec: str) -> int:
    """Camera index of a Linux device path: ``/dev/videoN`` or a link to one.

    Raises:
        CameraError: The path is missing or not a V4L2 capture node.
    """
    match = _DEV_VIDEO.match(spec)
    if match is None:
        target = _resolve_link(spec)
        match = _DEV_VIDEO.match(target)
        if match is None:
            if not os.path.exists(spec):
                raise CameraError(f"Video device not found: {spec} (is the camera plugged in?)")
            raise CameraError(f"Not a V4L2 video device: {spec} (resolves to {target})")
    return int(match.group(1))


def list_cameras(
    max_index: int = 4, api: str = "auto", skip: Collection[int] = ()
) -> list[CameraInfo]:
    """Probe camera indices ``0 … max_index - 1`` and return those that deliver frames.

    Each camera is opened briefly (its activity light may flash). This can take
    several seconds; call it from a background thread. A camera currently held
    by another application may be missing.

    ``skip`` lists indices that must not be probed, above all the camera this
    app's own worker is using: on DirectShow, probing an open camera "succeeds"
    and releasing the probe then stops the device under the worker.
    """
    pref = api_preference(api)
    excluded = {int(i) for i in skip}
    found: list[CameraInfo] = []
    for index in range(max(0, max_index)):
        if index in excluded:
            continue
        cap: cv2.VideoCapture | None = None
        try:
            cap = cv2.VideoCapture(index, pref)
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or not _valid_frame(frame):
                log.debug("Camera %d opened but returned no frame", index)
                continue
            h, w = frame.shape[:2]
            found.append(CameraInfo(index=index, name=f"Camera {index}", width=w, height=h))
        except cv2.error as exc:
            log.debug("Probing camera %d failed: %s", index, exc)
        finally:
            if cap is not None:
                cap.release()
    return found
