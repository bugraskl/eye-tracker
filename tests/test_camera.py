"""Tests for eye_tracker.vision.camera (no physical camera required)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from eye_tracker.vision import camera as camera_mod
from eye_tracker.vision.camera import (
    Camera,
    CameraError,
    CameraInfo,
    FileSource,
    FrameSource,
    api_preference,
    list_cameras,
    open_source,
)


# --------------------------------------------------------------------------- fakes
class FakeCapture:
    """Stand-in for cv2.VideoCapture driven by a script of read results."""

    def __init__(
        self,
        index: int | str = 0,
        api: int | None = None,
        *,
        opened: bool = True,
        reads: list[bool] | None = None,
        size: tuple[int, int] = (64, 48),
    ) -> None:
        self.index = index
        self.api = api
        self.opened = opened
        self.reads = list(reads) if reads is not None else []
        self.size = size
        self.props: dict[int, float] = {}
        self.grabs = 0
        self.read_calls = 0
        self.released = 0

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.props.get(prop, 0.0)

    def grab(self) -> bool:
        self.grabs += 1
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_calls += 1
        ok = self.reads.pop(0) if self.reads else True
        if not ok:
            return False, None
        w, h = self.size
        return True, np.full((h, w, 3), self.read_calls % 256, np.uint8)

    def release(self) -> None:
        self.released += 1

    def getBackendName(self) -> str:
        return "FAKE"


def install_capture(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> list[FakeCapture]:
    created: list[FakeCapture] = []

    def factory(index: int | str = 0, api: int | None = None) -> FakeCapture:
        cap = FakeCapture(index, api, **kwargs)  # type: ignore[arg-type]
        created.append(cap)
        return cap

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", factory)
    return created


# ------------------------------------------------------------------ api_preference
def test_api_preference_auto_matches_platform() -> None:
    expected = {
        "win32": cv2.CAP_DSHOW,
        "darwin": cv2.CAP_AVFOUNDATION,
        "linux": cv2.CAP_V4L2,
    }.get(sys.platform, cv2.CAP_ANY)
    assert api_preference("auto") == expected
    assert api_preference(" AUTO ") == expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("any", cv2.CAP_ANY),
        ("dshow", cv2.CAP_DSHOW),
        ("msmf", cv2.CAP_MSMF),
        ("avfoundation", cv2.CAP_AVFOUNDATION),
        ("v4l2", cv2.CAP_V4L2),
    ],
)
def test_api_preference_named(name: str, value: int) -> None:
    assert api_preference(name) == value


def test_api_preference_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown camera API"):
        api_preference("gstreamer-magic")


# ------------------------------------------------------------------------ Camera
def test_camera_is_a_frame_source() -> None:
    cam = Camera(0)
    assert isinstance(cam, FrameSource)
    assert not cam.is_open
    assert cam.read() is None
    assert cam.last_error is None


def test_camera_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="camera index"):
        Camera(-1)


def test_camera_open_configures_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_capture(monkeypatch, size=(320, 240))
    cam = Camera(2, width=320, height=240, api="any", fps=12)
    assert cam.open() is True
    assert cam.is_open
    assert cam.last_error is None
    assert cam.frame_size == (320, 240)
    cap = created[0]
    assert cap.index == 2
    assert cap.api == cv2.CAP_ANY
    mjpg = ord("M") | ord("J") << 8 | ord("P") << 16 | ord("G") << 24
    assert cap.props[cv2.CAP_PROP_FOURCC] == mjpg
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 320
    assert cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 240
    assert cap.props[cv2.CAP_PROP_FPS] == 12
    assert cap.props[cv2.CAP_PROP_BUFFERSIZE] == 1
    # Opening twice is a no-op.
    assert cam.open() is True
    assert len(created) == 1
    cam.release()
    assert cap.released == 1


def test_camera_open_failure_reports_error(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_capture(monkeypatch, opened=False)
    cam = Camera(3)
    assert cam.open() is False
    assert not cam.is_open
    assert cam.last_error is not None
    assert "Camera 3" in cam.last_error
    assert "another app" in cam.last_error
    assert created[0].released == 1
    assert cam.read() is None


def test_camera_open_without_frames_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_capture(monkeypatch, reads=[False] * 10)
    cam = Camera(1)
    assert cam.open() is False
    assert cam.last_error is not None
    assert "no frames" in cam.last_error
    assert created[0].released == 1


def test_camera_open_tolerates_empty_first_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    install_capture(monkeypatch, reads=[False, False, True])
    cam = Camera(0)
    assert cam.open() is True


def test_camera_open_when_constructor_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object) -> None:
        raise cv2.error("driver exploded")

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", boom)
    cam = Camera(0)
    assert cam.open() is False
    assert cam.last_error is not None


def test_camera_read_flushes_only_after_a_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_capture(monkeypatch)
    # fps=1 makes the "pause" window 1.5 s, so back-to-back reads never flush.
    cam = Camera(0, fps=1, flush=2)
    assert cam.open()
    cap = created[0]
    assert cam.read() is not None
    assert cap.grabs == 0
    cam._last_read -= 10.0  # pretend the reader paused for 10 s
    assert cam.read() is not None
    assert cap.grabs == 2


def test_camera_read_without_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_capture(monkeypatch)
    cam = Camera(0, flush=0)
    assert cam.open()
    cam._last_read -= 10.0
    assert cam.read() is not None
    assert created[0].grabs == 0


def test_camera_closes_after_consecutive_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    # One successful warm-up read, then failures.
    created = install_capture(monkeypatch, reads=[True] + [False] * 5)
    cam = Camera(4)
    assert cam.open()
    for _ in range(4):
        assert cam.read() is None
        assert cam.is_open
    assert cam.read() is None
    assert not cam.is_open
    assert cam.last_error is not None
    assert "stopped delivering frames" in cam.last_error
    assert created[0].released == 1
    assert cam.read() is None


def test_camera_failure_counter_resets_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    install_capture(monkeypatch, reads=[True, False, False, False, False, True, False, False])
    cam = Camera(0)
    assert cam.open()
    for _ in range(4):
        cam.read()
    assert cam.read() is not None
    cam.read()
    cam.read()
    assert cam.is_open


def test_camera_release_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_capture(monkeypatch)
    cam = Camera(0)
    cam.release()
    assert cam.open()
    cam.release()
    cam.release()
    assert created[0].released == 1
    assert not cam.is_open


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="AVFoundation may block on a camera permission prompt in headless runs",
)
def test_missing_real_camera_fails_gracefully() -> None:
    cam = Camera(63)
    assert cam.open() is False
    assert cam.last_error is not None
    assert "Camera 63" in cam.last_error
    assert cam.read() is None
    cam.release()


# ------------------------------------------------------------------ list_cameras
def test_list_cameras_reports_working_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeCapture] = []

    def factory(index: int = 0, api: int | None = None) -> FakeCapture:
        cap = FakeCapture(
            index,
            api,
            opened=index in (0, 2, 3),
            reads=[index != 3],  # camera 3 opens but delivers nothing
            size=(160 * (index + 1), 120 * (index + 1)),
        )
        created.append(cap)
        return cap

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", factory)
    found = list_cameras(max_index=4, api="any")
    assert found == [
        CameraInfo(index=0, name="Camera 0", width=160, height=120),
        CameraInfo(index=2, name="Camera 2", width=480, height=360),
    ]
    assert [c.index for c in created] == [0, 1, 2, 3]
    assert all(c.released == 1 for c in created)


# -------------------------------------------------------------------- FileSource
def _write_png(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", image)
    assert ok
    data.tofile(str(path))


def test_file_source_still_image(tmp_path: Path) -> None:
    image = np.zeros((60, 80, 3), np.uint8)
    image[10:20, 10:30] = (10, 200, 30)
    path = tmp_path / "still.png"
    _write_png(path, image)
    src = FileSource(path)
    assert isinstance(src, FrameSource)
    assert src.open()
    first = src.read()
    assert first is not None
    np.testing.assert_array_equal(first, image)
    first[:] = 255  # consumers may draw on frames; later reads must be unaffected
    second = src.read()
    assert second is not None
    np.testing.assert_array_equal(second, image)
    src.release()
    assert not src.is_open
    assert src.read() is None


def test_file_source_non_ascii_path(tmp_path: Path) -> None:
    path = tmp_path / "görüntü şğ.png"
    _write_png(path, np.full((12, 16, 3), 7, np.uint8))
    src = FileSource(path)
    assert src.open(), src.last_error
    frame = src.read()
    assert frame is not None
    assert frame.shape == (12, 16, 3)


def test_file_source_fits_bounding_box(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    _write_png(path, np.zeros((600, 800, 3), np.uint8))
    src = FileSource(path, width=400, height=400)
    assert src.open()
    frame = src.read()
    assert frame is not None
    assert frame.shape[:2] == (300, 400)  # aspect ratio preserved
    small = tmp_path / "small.png"
    _write_png(small, np.zeros((30, 40, 3), np.uint8))
    src = FileSource(small, width=640, height=480)
    assert src.open()
    frame = src.read()
    assert frame is not None
    assert frame.shape[:2] == (30, 40)  # never enlarged


def test_file_source_missing_file(tmp_path: Path) -> None:
    src = FileSource(tmp_path / "nope.mp4")
    assert src.open() is False
    assert src.last_error is not None
    assert "not found" in src.last_error
    assert src.read() is None


def test_file_source_undecodable_image(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not a png at all")
    src = FileSource(path)
    assert src.open() is False
    assert src.last_error is not None


LEVELS = (0, 50, 100, 150, 200, 250)


def _write_video(path: Path, levels: tuple[int, ...] = LEVELS, fps: float = 10.0) -> None:
    fourcc = ord("M") | ord("J") << 8 | ord("P") << 16 | ord("G") << 24
    writer = cv2.VideoWriter(str(path), cv2.CAP_OPENCV_MJPEG, fourcc, fps, (64, 48))
    if not writer.isOpened():
        pytest.skip("no video writer available in this OpenCV build")
    for level in levels:
        writer.write(np.full((48, 64, 3), level, np.uint8))
    writer.release()


def _level(frame: np.ndarray | None) -> int:
    assert frame is not None
    value = float(frame.mean())
    return min(LEVELS, key=lambda lv: abs(lv - value))


def test_file_source_video_loops(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    _write_video(path)
    src = FileSource(path)
    assert src.open(), src.last_error
    seen = [_level(src.read()) for _ in range(len(LEVELS) + 2)]
    assert seen == [*LEVELS, LEVELS[0], LEVELS[1]]
    src.release()
    assert src.read() is None


def test_file_source_video_realtime(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    _write_video(path, fps=10.0)
    src = FileSource(path, realtime=True)
    assert src.open(), src.last_error
    clock = [100.0]
    src._now = lambda: clock[0]
    assert _level(src.read()) == LEVELS[0]
    clock[0] += 0.05  # half a frame period later: still the same frame
    assert _level(src.read()) == LEVELS[0]
    clock[0] += 0.26  # 0.31 s after the start -> frame 3
    assert _level(src.read()) == LEVELS[3]
    clock[0] += 0.4  # 0.71 s: frame 7 of a 6-frame clip -> wrapped to frame 1
    assert _level(src.read()) == LEVELS[1]


# ------------------------------------------------------------------- open_source
def test_open_source_camera_index() -> None:
    src = open_source(" 1 ", 320, 240, "any")
    assert isinstance(src, Camera)
    assert src.index == 1
    assert (src.width, src.height) == (320, 240)
    assert not src.is_open  # created, not opened


def test_open_source_file(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    _write_png(path, np.zeros((10, 10, 3), np.uint8))
    src = open_source(str(path), 640, 480, "auto")
    assert isinstance(src, FileSource)
    assert src.realtime


@pytest.mark.parametrize("device", ["", "   ", "definitely/not/here.mp4"])
def test_open_source_errors(device: str) -> None:
    with pytest.raises(CameraError):
        open_source(device, 640, 480, "auto")


def test_open_source_bad_api() -> None:
    with pytest.raises(CameraError, match="Unknown camera API"):
        open_source("0", 640, 480, "bogus")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/dev/videoN is Linux only")
def test_open_source_dev_video() -> None:
    src = open_source("/dev/video2", 640, 480, "auto")
    assert isinstance(src, Camera)
    assert src.index == 2
