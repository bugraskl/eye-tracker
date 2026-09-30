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
    device_index,
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
        props: dict[int, float] | None = None,
        grab_ok: bool = True,
    ) -> None:
        self.index = index
        self.api = api
        self.opened = opened
        self.reads = list(reads) if reads is not None else []
        self.size = size
        self.props: dict[int, float] = dict(props or {})
        self.set_order: list[int] = []
        self.grab_ok = grab_ok
        self.grabs = 0
        self.read_calls = 0
        self.released = 0

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        self.set_order.append(prop)
        return True

    def get(self, prop: int) -> float:
        return self.props.get(prop, 0.0)

    def grab(self) -> bool:
        self.grabs += 1
        return self.grab_ok

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
    # A camera blocked by the OS privacy switch fails the same way.
    assert "privacy settings" in cam.last_error
    assert created[0].released == 1
    assert cam.read() is None


MJPG = ord("M") | ord("J") << 8 | ord("P") << 16 | ord("G") << 24


def test_dshow_sets_only_what_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every DirectShow property change rebuilds the capture graph."""
    current = {
        cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        cv2.CAP_PROP_FPS: 30.0,
    }
    created = install_capture(monkeypatch, props=current, size=(640, 480))
    cam = Camera(0, width=640, height=480, api="dshow", fps=15)
    assert cam.open()
    cap = created[0]
    assert cap.set_order == [cv2.CAP_PROP_FPS, cv2.CAP_PROP_BUFFERSIZE]
    assert cv2.CAP_PROP_FOURCC not in cap.props  # uncompressed 640x480@15 fits USB 2


def test_dshow_asks_for_mjpg_last_at_high_bandwidth(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_capture(monkeypatch, size=(1280, 720))
    cam = Camera(0, width=1280, height=720, api="dshow", fps=15)
    assert cam.open()
    cap = created[0]
    assert cap.set_order == [
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
        cv2.CAP_PROP_FOURCC,  # last: DirectShow keeps size and rate when only the format changes
        cv2.CAP_PROP_BUFFERSIZE,
    ]
    assert cap.props[cv2.CAP_PROP_FOURCC] == MJPG


def test_other_apis_set_the_format_first(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_capture(monkeypatch)
    assert Camera(0, api="v4l2").open()
    assert created[0].set_order[0] == cv2.CAP_PROP_FOURCC


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


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_camera_gives_up_after_failing_for_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drivers whose reads block for seconds must not keep the worker for a minute."""
    clock = FakeClock()
    monkeypatch.setattr(camera_mod.time, "monotonic", clock)
    created = install_capture(monkeypatch, reads=[True] + [False] * 10)
    cam = Camera(0)
    assert cam.open()
    clock.now += 0.05
    assert cam.read() is None  # first failure
    clock.now += 2.0  # a read that blocked for 2 s
    assert cam.read() is None
    assert cam.is_open
    clock.now += 2.0
    assert cam.read() is None
    assert not cam.is_open  # 4 s of failures, only 3 reads
    assert cam.last_error is not None
    assert "stopped delivering frames" in cam.last_error
    assert created[0].released == 1


def test_camera_does_not_flush_while_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(camera_mod.time, "monotonic", clock)
    created = install_capture(monkeypatch, reads=[True, False, False, True])
    cam = Camera(0, fps=10, flush=1)
    assert cam.open()
    cap = created[0]
    clock.now += 1.0  # paused: the next read flushes
    assert cam.read() is None
    assert cap.grabs == 1
    clock.now += 1.0  # paused again, but failing: no extra blocking grab
    assert cam.read() is None
    assert cap.grabs == 1
    clock.now += 0.01
    assert cam.read() is not None


def test_failed_flush_counts_as_a_failed_read(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(camera_mod.time, "monotonic", clock)
    created = install_capture(monkeypatch, grab_ok=False)
    cam = Camera(0, fps=10, flush=2)
    assert cam.open()
    cap = created[0]
    reads_after_open = cap.read_calls
    clock.now += 1.0
    assert cam.read() is None
    assert cap.grabs == 1  # stopped at the first failed grab
    assert cap.read_calls == reads_after_open  # and did not block in read() as well


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


def test_list_cameras_never_touches_skipped_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probing the worker's own DirectShow camera would stop it under the worker."""
    created = install_capture(monkeypatch)
    found = list_cameras(max_index=3, api="any", skip={0})
    assert [c.index for c in created] == [1, 2]
    assert [c.index for c in found] == [1, 2]


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


def test_open_source_records_the_device_string(tmp_path: Path) -> None:
    cam = open_source(" 2 ", 640, 480, "any")
    assert isinstance(cam, Camera)
    assert cam.device == "2"
    path = tmp_path / "frame.png"
    _write_png(path, np.zeros((10, 10, 3), np.uint8))
    src = open_source(str(path), 640, 480, "auto")
    assert getattr(src, "device", None) == str(path)


@pytest.mark.parametrize(
    "device",
    [
        "rtsp://192.168.1.20/stream",
        "http://example.com/clip.mp4",
        "HTTPS://example.com/clip.mp4",
        "https:example.com/clip.mp4",
        "tcp:10.0.0.1:5000",
        "udp://@:1234",
        "srt://host:9000",
        "concat:a.mp4|b.mp4",
        "file:clip.mp4",
        "rtmp+tls:host",
        "subfile,,start,0,end,0,,:video.mp4",
        "C:/videos/../x://y",
    ],
)
def test_open_source_refuses_network_and_protocol_sources(device: str) -> None:
    """Offline guarantee: FFmpeg must never be handed a URL or a protocol."""
    with pytest.raises(CameraError, match="not supported"):
        open_source(device, 640, 480, "auto")


@pytest.mark.parametrize("device", ["rtsp://camera/stream", "tcp:10.0.0.1:5000"])
def test_file_source_refuses_urls(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    created = install_capture(monkeypatch)
    src = FileSource(device)
    assert src.open() is False
    assert src.last_error is not None
    assert "network sources are not supported" in src.last_error
    assert created == []


def test_windows_drive_paths_are_files(tmp_path: Path) -> None:
    missing = (
        "C:\\definitely\\missing.mp4" if sys.platform == "win32" else "/definitely/missing.mp4"
    )
    with pytest.raises(CameraError, match="not found"):
        open_source(missing, 640, 480, "auto")
    path = tmp_path / "clip.png"
    _write_png(path, np.zeros((10, 10, 3), np.uint8))
    assert isinstance(open_source(str(path.resolve()), 640, 480, "auto"), FileSource)


def test_file_source_reads_videos_with_ffmpeg_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Other capture backends (GStreamer, Media Foundation) follow playlist URLs."""
    path = tmp_path / "clip.avi"
    path.write_bytes(b"\x00" * 64)
    created = install_capture(monkeypatch)
    src = FileSource(path)
    assert src.open()
    assert created
    assert all(cap.api == cv2.CAP_FFMPEG for cap in created)


def test_local_playlist_cannot_reach_the_network(tmp_path: Path) -> None:
    """FFmpeg's whitelist stops a local playlist from fetching remote segments."""
    import socket
    import threading

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    server.settimeout(0.1)
    port = server.getsockname()[1]
    hits: list[int] = []
    stop = threading.Event()

    def accept() -> None:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            hits.append(1)
            conn.close()

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    playlist = tmp_path / "list.m3u8"
    playlist.write_text(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1.0,\n"
        f"http://127.0.0.1:{port}/segment.ts\n#EXT-X-ENDLIST\n"
    )
    try:
        src = FileSource(playlist)
        src.open()
        src.release()
    finally:
        stop.set()
        thread.join(1.0)
        server.close()
    assert hits == []


# ------------------------------------------------------------------ Linux devices
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/dev/videoN is Linux only")
def test_open_source_dev_video() -> None:
    src = open_source("/dev/video2", 640, 480, "auto")
    assert isinstance(src, Camera)
    assert src.index == 2
    assert src.device == "/dev/video2"


@pytest.fixture
def linux(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Pretend to run on Linux with a fake /dev (link -> target)."""
    links: dict[str, str] = {}
    real_exists, real_isfile = camera_mod.os.path.exists, camera_mod.os.path.isfile

    def exists(path: str) -> bool:
        return path in links if str(path).startswith("/dev/") else real_exists(path)

    def isfile(path: str) -> bool:
        return False if str(path).startswith("/dev/") else real_isfile(path)

    monkeypatch.setattr(camera_mod.sys, "platform", "linux")
    monkeypatch.setattr(camera_mod, "_resolve_link", lambda p: links.get(p, p))
    monkeypatch.setattr(camera_mod.os.path, "exists", exists)
    monkeypatch.setattr(camera_mod.os.path, "isfile", isfile)
    return links


def test_stable_v4l_links_resolve_to_the_camera(linux: dict[str, str]) -> None:
    link = "/dev/v4l/by-id/usb-Logitech_C920_1234-video-index0"
    linux[link] = "/dev/video7"
    src = open_source(link, 640, 480, "auto")
    assert isinstance(src, Camera)
    assert src.index == 7
    assert src.device == link  # the stable name, for calibrations and settings
    linux[link] = "/dev/video2"  # re-plugged: resolved again on the next open
    again = open_source(link, 640, 480, "auto")
    assert isinstance(again, Camera)
    assert again.index == 2


def test_missing_and_non_video_devices(linux: dict[str, str]) -> None:
    with pytest.raises(CameraError, match="not found"):
        open_source("/dev/v4l/by-id/usb-unplugged-video-index0", 640, 480, "auto")
    linux["/dev/v4l/by-id/weird"] = "/dev/snd/pcmC0D0c"
    with pytest.raises(CameraError, match="Not a V4L2 video device"):
        open_source("/dev/v4l/by-id/weird", 640, 480, "auto")


def test_device_index_follows_stable_links(linux: dict[str, str]) -> None:
    """What the settings dialog must not probe: the index the device setting leads to."""
    link = "/dev/v4l/by-id/usb-Logitech_C920_1234-video-index0"
    linux[link] = "/dev/video3"
    assert device_index(" 1 ") == 1
    assert device_index("/dev/video5") == 5
    assert device_index(link) == 3
    # Unplugged, or not a camera: no index, and no exception either.
    assert device_index("/dev/v4l/by-id/usb-unplugged-video-index0") is None
    linux["/dev/v4l/by-id/weird"] = "/dev/snd/pcmC0D0c"
    assert device_index("/dev/v4l/by-id/weird") is None
    assert device_index("clip.mp4") is None  # a video file


def test_device_paths_are_files_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_mod.sys, "platform", "win32")
    assert device_index("/dev/video5") is None
    assert device_index("2") == 2


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_real_symlink_to_a_video_node(tmp_path: Path) -> None:
    link = tmp_path / "by-id-camera"
    link.symlink_to("/dev/video7")
    assert camera_mod._v4l2_index(str(link)) == 7


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_v4l2_select_timeout_is_shortened() -> None:
    import os

    assert os.environ.get("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT") is not None
