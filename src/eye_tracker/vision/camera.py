"""Frame sources: webcams through OpenCV and video/image files for tests and demos.

Every source hands out BGR ``uint8`` frames that live only in memory. Nothing in
this module writes image data anywhere.

Sources are used from the vision worker thread. ``open()`` may block for a few
seconds (some Windows camera drivers are slow to start) but never raises;
failures are reported through ``last_error``.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
import sys
import time
from collections.abc import Callable
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

# A camera that fails this many reads in a row is considered gone (unplugged,
# taken over by another application or its driver crashed).
_MAX_READ_FAILURES = 5
# Reads attempted while opening; the first frames of some cameras are empty.
_WARMUP_READS = 5

_API_BY_NAME: dict[str, int] = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "v4l2": cv2.CAP_V4L2,
}

_DEV_VIDEO = re.compile(r"^/dev/video(\d+)$")


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
        self._api_pref = api_preference(api)
        self._cap: cv2.VideoCapture | None = None
        self._error: str | None = None
        self._failures = 0
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
        started = time.monotonic()
        try:
            cap = cv2.VideoCapture(self.index, self._api_pref)
        except Exception as exc:  # open() must never raise; drivers fail in odd ways
            log.debug("VideoCapture(%d) raised: %s", self.index, exc)
            self._error = f"{self.name} could not be opened — is another app using it?"
            return False
        if not cap.isOpened():
            cap.release()
            self._error = f"{self.name} could not be opened — is another app using it?"
            log.debug("%s did not open (%.0f ms)", self.name, (time.monotonic() - started) * 1e3)
            return False

        # FOURCC first: on DirectShow and V4L2 changing the pixel format afterwards
        # can reset the frame size. MJPG lets USB 2 cameras deliver 640x480 at full
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
            self._error = f"{self.name} opened but delivers no frames — is another app using it?"
            log.debug("%s delivered no frames", self.name)
            return False

        self._cap = cap
        self._error = None
        self._last_read = time.monotonic()
        self._frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        try:
            backend = cap.getBackendName()
        except cv2.error:
            backend = "?"
        log.debug(
            "%s opened via %s at %dx%d in %.0f ms",
            self.name,
            backend,
            self._frame_size[0],
            self._frame_size[1],
            (time.monotonic() - started) * 1e3,
        )
        return True

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
            if self.flush and time.monotonic() - self._last_read > 1.5 / self.fps:
                for _ in range(self.flush):
                    cap.grab()
            ok, frame = cap.read()
        except Exception as exc:
            log.debug("%s read raised: %s", self.name, exc)
        self._last_read = time.monotonic()

        if not ok or not _valid_frame(frame):
            self._failures += 1
            if self._failures >= _MAX_READ_FAILURES:
                self._error = (
                    f"{self.name} stopped delivering frames — was it unplugged or taken "
                    "by another app?"
                )
                log.debug("%s stopped delivering frames; closing it", self.name)
                self.release()
            return None
        self._failures = 0
        return frame

    def release(self) -> None:
        cap, self._cap = self._cap, None
        self._failures = 0
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
            cap = cv2.VideoCapture(str(self.path))
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

    ``"0"``, ``"1"``, … select a camera by index (``/dev/videoN`` is accepted on
    Linux); anything else is a path to a video or image file.

    Raises:
        CameraError: The file does not exist or the settings are invalid.
    """
    spec = device.strip()
    if not spec:
        raise CameraError("No camera configured")
    index: int | None = None
    if spec.isdigit():
        index = int(spec)
    elif sys.platform.startswith("linux") and (match := _DEV_VIDEO.match(spec)):
        index = int(match.group(1))
    if index is not None:
        try:
            return Camera(index, width, height, api, fps=fps)
        except ValueError as exc:
            raise CameraError(str(exc)) from exc

    path = Path(spec).expanduser()
    if not path.is_file():
        raise CameraError(f"Video source not found: {path}")
    return FileSource(path, width, height, realtime=realtime)


def list_cameras(max_index: int = 4, api: str = "auto") -> list[CameraInfo]:
    """Probe camera indices ``0 … max_index - 1`` and return those that deliver frames.

    Each camera is opened briefly (its activity light may flash). This can take
    several seconds; call it from a background thread. A camera currently held
    by another application (or by this app's own worker) may be missing.
    """
    pref = api_preference(api)
    found: list[CameraInfo] = []
    for index in range(max(0, max_index)):
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
