"""Tests for the macOS platform layer.

pyobjc is replaced by small fakes, so these run on every OS. Nothing here
locks the screen, sleeps the displays or moves the real cursor: the ctypes
loader and every subprocess entry point are stubbed by an autouse fixture.
"""

from __future__ import annotations

import json
import logging
import math
import plistlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eye_tracker.platform import macos
from eye_tracker.platform.base import PlatformServices
from eye_tracker.types import Rect, WindowRef

PYOBJC_MODULES = (
    "Quartz",
    "AppKit",
    "ApplicationServices",
    "CoreFoundation",
    "Foundation",
    "AVFoundation",
    "objc",
)
PID = 321
OTHER_PID = 654
AX_ERROR_ATTRIBUTE_UNSUPPORTED = -25205
AX_ERROR_CANNOT_COMPLETE = -25204


class Clock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


class Recorder:
    """Stands in for ``subprocess.run`` / ``subprocess.Popen`` and the tool lookup."""

    def __init__(self) -> None:
        self.tools: dict[str, str] = {}
        self.returncode = 0
        self.runs: list[list[str]] = []
        self.popens: list[list[str]] = []

    def tool(self, name: str) -> str | None:
        return self.tools.get(name)

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert kwargs.get("timeout"), "tools must run with a timeout"
        self.runs.append(list(argv))
        return subprocess.CompletedProcess(argv, self.returncode, b"", b"")

    def popen(self, argv: list[str], **kwargs: Any) -> Any:
        self.popens.append(list(argv))
        return SimpleNamespace(wait=lambda timeout=None: 0)


@pytest.fixture(autouse=True)
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(macos, "_tool", rec.tool)
    monkeypatch.setattr(macos.subprocess, "run", rec.run)
    monkeypatch.setattr(macos.subprocess, "Popen", rec.popen)

    def no_library(path: str) -> Any:
        raise OSError(f"{path} is not loadable in tests")

    monkeypatch.setattr(macos, "_load_library", no_library)
    monkeypatch.setattr(macos, "_IS_MACOS", False)
    return rec


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "macos-accessibility.json"


@pytest.fixture
def plat(clock: Clock, state_path: Path) -> macos.MacPlatform:
    """A MacPlatform with every pyobjc module "missing" until a test injects a fake."""
    platform = macos.MacPlatform(clock=clock, state_path=state_path)
    platform._modules.update(dict.fromkeys(PYOBJC_MODULES))
    return platform


# ---------------------------------------------------------------------- fakes
class FakeQuartz:
    kCGEventSourceStateCombinedSessionState = 0
    kCGAnyInputEventType = 0xFFFFFFFF
    kCGEventKeyDown = 10
    kCGWindowListOptionOnScreenOnly = 1
    kCGWindowListExcludeDesktopElements = 16
    kCGNullWindowID = 0

    def __init__(self) -> None:
        self.idle: dict[int, float] = {0xFFFFFFFF: 12.5, 10: 3.25}
        self.session: dict[str, Any] | None = {}
        self.warp_error = 0
        self.warps: list[tuple[float, float]] = []
        self.associations: list[bool] = []
        self.windows: list[dict[str, Any]] = []

    def CGEventSourceSecondsSinceLastEventType(self, state: int, event_type: int) -> float:
        assert state == 0
        return self.idle[event_type]

    def CGSessionCopyCurrentDictionary(self) -> dict[str, Any] | None:
        return self.session

    def CGWarpMouseCursorPosition(self, point: tuple[float, float]) -> int:
        self.warps.append(point)
        return self.warp_error

    def CGAssociateMouseAndMouseCursorPosition(self, connected: bool) -> int:
        self.associations.append(connected)
        return 0

    def CGWindowListCopyWindowInfo(self, options: int, relative_to: int) -> list[dict[str, Any]]:
        assert options == 1 | 16
        assert relative_to == 0
        return self.windows


def cg_window(
    pid: int, x: float, y: float, w: float, h: float, *, layer: int = 0, alpha: float = 1.0
) -> dict[str, Any]:
    return {
        "kCGWindowOwnerPID": pid,
        "kCGWindowLayer": layer,
        "kCGWindowAlpha": alpha,
        "kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h},
    }


class AXValue:
    """Mimics an AXValueRef holding a CGPoint or a CGSize."""

    def __init__(self, kind: int, a: float, b: float) -> None:
        self.kind, self.a, self.b = kind, a, b

    def __str__(self) -> str:
        if self.kind == 1:
            return f"<AXValue 0x1> {{value = x:{self.a:.6f} y:{self.b:.6f} type = kAXValueCGPoint}}"
        return f"<AXValue 0x2> {{value = w:{self.a:.6f} h:{self.b:.6f} type = kAXValueCGSize}}"


class FakeAX:
    kAXValueCGPointType = 1
    kAXValueCGSizeType = 2

    def __init__(self) -> None:
        self.trusted = True
        self.attrs: dict[tuple[Any, str], Any] = {}
        self.timeouts: list[tuple[Any, float]] = []
        self.sets: list[tuple[Any, str, Any]] = []
        self.actions: list[tuple[Any, str]] = []
        self.prompts: list[dict[str, Any]] = []
        self.hit: Any = None
        self.pids: dict[Any, int] = {}
        self.unpack = True
        #: Elements whose app does not answer (every query times out).
        self.hung: set[Any] = set()
        self.queries: list[tuple[Any, str]] = []

    # element factories
    def AXUIElementCreateApplication(self, pid: int) -> str:
        return f"app:{pid}"

    def AXUIElementCreateSystemWide(self) -> str:
        return "system"

    # queries
    def AXUIElementCopyAttributeValue(self, element: Any, name: str, out: Any) -> tuple[int, Any]:
        assert out is None
        key = (element, name)
        self.queries.append(key)
        if element in self.hung:
            return (AX_ERROR_CANNOT_COMPLETE, None)
        if key not in self.attrs:
            return (AX_ERROR_ATTRIBUTE_UNSUPPORTED, None)
        return (0, self.attrs[key])

    def AXValueGetValue(self, value: AXValue, value_type: int, out: Any) -> tuple[bool, Any]:
        assert out is None
        if not self.unpack or value.kind != value_type:
            return (False, None)
        if value_type == 1:
            return (True, SimpleNamespace(x=value.a, y=value.b))
        return (True, SimpleNamespace(width=value.a, height=value.b))

    def AXUIElementCopyElementAtPosition(
        self, system: Any, x: float, y: float, out: Any
    ) -> tuple[int, Any]:
        assert system == "system"
        assert isinstance(x, float)
        return (0, self.hit) if self.hit is not None else (-25200, None)

    def AXUIElementGetPid(self, element: Any, out: Any) -> tuple[int, int]:
        return (0, self.pids[element])

    # actions
    def AXUIElementSetMessagingTimeout(self, element: Any, seconds: float) -> int:
        self.timeouts.append((element, seconds))
        return 0

    def AXUIElementSetAttributeValue(self, element: Any, name: str, value: Any) -> int:
        self.sets.append((element, name, value))
        return 0

    def AXUIElementPerformAction(self, element: Any, action: str) -> int:
        self.actions.append((element, action))
        return 0

    def AXIsProcessTrusted(self) -> bool:
        return self.trusted

    def AXIsProcessTrustedWithOptions(self, options: dict[str, Any]) -> bool:
        self.prompts.append(dict(options))
        return self.trusted

    # helpers for tests
    def window(self, name: str, rect: tuple[float, float, float, float], **attrs: Any) -> str:
        x, y, w, h = rect
        self.attrs[(name, "AXPosition")] = AXValue(1, x, y)
        self.attrs[(name, "AXSize")] = AXValue(2, w, h)
        self.attrs[(name, "AXRole")] = "AXWindow"
        self.attrs[(name, "AXMinimized")] = attrs.pop("minimized", False)
        for key, value in attrs.items():
            self.attrs[(name, key)] = value
        return name


class FakeRunningApp:
    def __init__(self, pid: int, *, terminated: bool = False, hidden: bool = False) -> None:
        self.pid = pid
        self.terminated = terminated
        self.hidden = hidden
        self.activations: list[int] = []
        self.activate_result = True

    def processIdentifier(self) -> int:
        return self.pid

    def isTerminated(self) -> bool:
        return self.terminated

    def isHidden(self) -> bool:
        return self.hidden

    def activateWithOptions_(self, options: int) -> bool:
        self.activations.append(options)
        return self.activate_result


class FakeAppKit:
    NSApplicationActivateIgnoringOtherApps = 2
    NSApplicationActivationPolicyAccessory = 1

    def __init__(self, frontmost: int = PID) -> None:
        self.apps: dict[int, FakeRunningApp] = {PID: FakeRunningApp(PID)}
        self.frontmost = frontmost
        self.policies: list[int] = []
        kit = self

        class Workspace:
            @staticmethod
            def sharedWorkspace() -> Any:
                return SimpleNamespace(frontmostApplication=lambda: kit.apps.get(kit.frontmost))

        class RunningApplication:
            @staticmethod
            def runningApplicationWithProcessIdentifier_(pid: int) -> FakeRunningApp | None:
                return kit.apps.get(pid)

        class Application:
            @staticmethod
            def sharedApplication() -> Any:
                def set_policy(policy: int) -> bool:
                    kit.policies.append(policy)
                    return True

                return SimpleNamespace(setActivationPolicy_=set_policy)

        self.NSWorkspace = Workspace
        self.NSRunningApplication = RunningApplication
        self.NSApplication = Application


@pytest.fixture
def quartz(plat: macos.MacPlatform) -> FakeQuartz:
    fake = FakeQuartz()
    plat._modules["Quartz"] = fake
    return fake


@pytest.fixture
def ax(plat: macos.MacPlatform) -> FakeAX:
    fake = FakeAX()
    plat._modules["ApplicationServices"] = fake
    return fake


@pytest.fixture
def appkit(plat: macos.MacPlatform) -> FakeAppKit:
    fake = FakeAppKit()
    plat._modules["AppKit"] = fake
    return fake


# ------------------------------------------------------------ graceful degradation
def test_everything_degrades_without_pyobjc(plat: macos.MacPlatform, recorder: Recorder) -> None:
    ref = WindowRef(handle=(PID, None), pid=PID)
    assert plat.seconds_since_input() is None
    assert plat.seconds_since_key_input() is None
    assert plat.is_session_locked() is None
    assert plat.move_cursor(1, 2) is None  # caller falls back to QCursor
    assert plat.foreground_window() is None
    assert plat.window_at(1, 2) is None
    assert plat.activate_window(ref) is False
    assert plat.is_window_valid(ref) is False
    assert plat.window_rect(ref) is None
    assert plat.same_window(ref, None) is False
    assert plat.camera_in_use_by_other_app() is None
    assert plat.permissions() == {"camera": None, "accessibility": None}
    assert plat.set_accessory_app() is False
    plat.request_permission("accessibility")
    plat.request_permission("camera")
    plat.prepare_process()
    assert set(plat.capabilities()) == set(PlatformServices().capabilities())
    assert recorder.runs == []
    assert recorder.popens == []


def test_real_import_failure_is_cached(monkeypatch: pytest.MonkeyPatch, clock: Clock) -> None:
    attempts: list[str] = []

    def failing_import(name: str) -> Any:
        attempts.append(name)
        raise ImportError(name)

    monkeypatch.setattr(macos, "importlib", SimpleNamespace(import_module=failing_import))
    platform = macos.MacPlatform(clock=clock)
    assert platform.seconds_since_input() is None
    assert platform.seconds_since_input() is None
    assert attempts == ["Quartz"]


def test_power_actions_are_macos_only(plat: macos.MacPlatform, recorder: Recorder) -> None:
    recorder.tools = {"pmset": "/usr/bin/pmset", "caffeinate": "/usr/bin/caffeinate"}
    assert plat.lock_screen() is False
    assert plat.display_off() is False
    assert plat.wake_display() is False
    assert plat.open_permission_settings("camera") is False
    assert recorder.runs == []
    assert recorder.popens == []


# ------------------------------------------------------------------ lock / power
class FakeLockFunc:
    def __init__(self, status: int) -> None:
        self.status = status
        self.calls = 0
        self.restype: Any = None
        self.argtypes: Any = None

    def __call__(self) -> int:
        self.calls += 1
        return self.status


@pytest.fixture
def on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos, "_IS_MACOS", True)


def _login_framework(monkeypatch: pytest.MonkeyPatch, func: FakeLockFunc) -> list[str]:
    loaded: list[str] = []

    def load(path: str) -> Any:
        loaded.append(path)
        return SimpleNamespace(SACLockScreenImmediate=func)

    monkeypatch.setattr(macos, "_load_library", load)
    return loaded


@pytest.mark.usefixtures("on_macos")
def test_lock_uses_login_framework(
    monkeypatch: pytest.MonkeyPatch, plat: macos.MacPlatform, recorder: Recorder
) -> None:
    func = FakeLockFunc(0)
    loaded = _login_framework(monkeypatch, func)
    recorder.tools = {"pmset": "/usr/bin/pmset"}
    assert plat.lock_screen() is True
    assert loaded == [macos.LOGIN_FRAMEWORK]
    assert func.calls == 1
    assert func.argtypes == []
    assert recorder.runs == []


@pytest.mark.usefixtures("on_macos")
def test_lock_falls_back_to_display_sleep(
    monkeypatch: pytest.MonkeyPatch, plat: macos.MacPlatform, recorder: Recorder
) -> None:
    _login_framework(monkeypatch, FakeLockFunc(-1))
    recorder.tools = {"pmset": "/usr/bin/pmset"}
    assert plat.lock_screen() is True
    assert recorder.runs == [["/usr/bin/pmset", "displaysleepnow"]]


@pytest.mark.usefixtures("on_macos")
def test_lock_fails_without_any_mechanism(plat: macos.MacPlatform, recorder: Recorder) -> None:
    assert plat.lock_screen() is False
    recorder.tools = {"pmset": "/usr/bin/pmset"}
    recorder.returncode = 1
    assert plat.lock_screen() is False


@pytest.mark.usefixtures("on_macos")
def test_display_off_and_wake(plat: macos.MacPlatform, recorder: Recorder) -> None:
    recorder.tools = {"pmset": "/usr/bin/pmset", "caffeinate": "/usr/bin/caffeinate"}
    assert plat.display_off() is True
    assert plat.wake_display() is True
    assert recorder.runs == [["/usr/bin/pmset", "displaysleepnow"]]
    assert recorder.popens == [["/usr/bin/caffeinate", "-u", "-t", "2"]]
    recorder.tools = {}
    assert plat.wake_display() is False


# ------------------------------------------------------------------ input / session
def test_idle_times_from_quartz(plat: macos.MacPlatform, quartz: FakeQuartz) -> None:
    assert plat.seconds_since_input() == pytest.approx(12.5)
    assert plat.seconds_since_key_input() == pytest.approx(3.25)


def test_idle_uses_fallback_constants(plat: macos.MacPlatform) -> None:
    calls: list[tuple[int, int]] = []

    def since(state: int, event_type: int) -> float:
        calls.append((state, event_type))
        return 1.0

    plat._modules["Quartz"] = SimpleNamespace(CGEventSourceSecondsSinceLastEventType=since)
    assert plat.seconds_since_input() == 1.0
    assert plat.seconds_since_key_input() == 1.0
    assert calls == [(0, 0xFFFFFFFF), (0, 10)]


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_idle_rejects_nonsense(plat: macos.MacPlatform, quartz: FakeQuartz, value: float) -> None:
    quartz.idle[0xFFFFFFFF] = value
    assert plat.seconds_since_input() is None


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (None, None),
        ({}, False),
        ({"CGSSessionScreenIsLocked": True}, True),
        ({"CGSSessionScreenIsLocked": 1, "kCGSSessionOnConsoleKey": True}, True),
        ({"CGSSessionScreenIsLocked": False, "kCGSSessionOnConsoleKey": True}, False),
        ({"kCGSSessionOnConsoleKey": False}, True),  # fast user switching
    ],
)
def test_session_locked(
    plat: macos.MacPlatform, quartz: FakeQuartz, info: dict[str, Any] | None, expected: bool | None
) -> None:
    quartz.session = info
    assert plat.is_session_locked() is expected


def test_move_cursor_warps_and_reassociates(plat: macos.MacPlatform, quartz: FakeQuartz) -> None:
    assert plat.move_cursor(2560, -300) is True
    assert quartz.warps == [(2560.0, -300.0)]
    assert quartz.associations == [True]
    quartz.warp_error = 1001
    assert plat.move_cursor(0, 0) is False


# ------------------------------------------------------------------ pure helpers
def test_err_value_and_handle_splitting() -> None:
    assert macos._err_value((0, "x")) == (0, "x")
    assert macos._err_value((-25202, None)) == (-25202, None)
    assert macos._err_value((None, "x")) == (0, "x")
    assert macos._err_value("bare") == (0, "bare")
    assert macos._split_handle((PID, "win")) == (PID, "win")
    assert macos._split_handle((PID, None)) == (PID, None)
    assert macos._split_handle((0, "win")) == (None, None)
    assert macos._split_handle((True, "win")) == (None, None)
    assert macos._split_handle(12345) == (None, None)


def test_ax_value_repr_parsing() -> None:
    point = "<AXValue 0x6000> {value = x:24.000000 y:-38.500000 type = kAXValueCGPointType}"
    size = "<AXValue 0x6001> {value = w:1200.000000 h:800.000000 type = kAXValueCGSizeType}"
    assert macos._pair_from_repr(point, size=False) == (24.0, -38.5)
    assert macos._pair_from_repr(size, size=True) == (1200.0, 800.0)
    assert macos._pair_from_repr(point, size=True) is None
    assert macos._pair_from_repr("garbage", size=False) is None


def test_rect_from_pairs() -> None:
    assert macos._rect_from((10.4, 20.6), (800.0, 600.0)) == Rect(10, 21, 800, 600)
    assert macos._rect_from(None, (1.0, 1.0)) is None
    assert macos._rect_from((0.0, 0.0), (0.0, 10.0)) is None
    assert macos._rect_from((math.nan, 0.0), (10.0, 10.0)) is None


def test_ax_rect_unpacks_values(ax: FakeAX) -> None:
    ax.window("win", (-1440, 25, 1200, 800))
    assert macos._ax_rect(ax, "win") == Rect(-1440, 25, 1200, 800)
    assert macos._ax_rect(ax, "missing") is None
    assert macos._ax_rect(None, "win") is None


def test_ax_rect_falls_back_to_description(ax: FakeAX) -> None:
    ax.window("win", (10, 20, 300, 200))
    ax.unpack = False  # AXValueGetValue cannot unpack in this pyobjc build
    assert macos._ax_rect(ax, "win") == Rect(10, 20, 300, 200)


def test_ax_window_of_walks_up_the_hierarchy(ax: FakeAX) -> None:
    ax.window("win", (0, 0, 10, 10))
    ax.attrs[("button", "AXRole")] = "AXButton"
    ax.attrs[("button", "AXWindow")] = "win"
    ax.attrs[("cell", "AXRole")] = "AXCell"
    ax.attrs[("cell", "AXParent")] = "group"
    ax.attrs[("group", "AXRole")] = "AXGroup"
    ax.attrs[("group", "AXParent")] = "win"
    ax.attrs[("menu", "AXRole")] = "AXMenuItem"
    ax.attrs[("menu", "AXParent")] = "app"
    ax.attrs[("app", "AXRole")] = "AXApplication"
    assert macos._ax_window_of(ax, "win") == "win"
    assert macos._ax_window_of(ax, "button") == "win"
    assert macos._ax_window_of(ax, "cell") == "win"
    assert macos._ax_window_of(ax, "menu") is None


def test_ax_window_of_is_bounded(ax: FakeAX) -> None:
    ax.attrs[("loop", "AXRole")] = "AXGroup"
    ax.attrs[("loop", "AXParent")] = "loop"  # a broken, cyclic hierarchy
    assert macos._ax_window_of(ax, "loop") is None


def test_cg_window_selection() -> None:
    windows = [
        cg_window(OTHER_PID, 0, 0, 3000, 25, layer=25),  # menu bar
        cg_window(999, 0, 0, 3000, 2000),  # our own window (pid 999 below)
        cg_window(OTHER_PID, 100, 100, 400, 300, alpha=0.0),  # invisible
        cg_window(PID, 50, 50, 800, 600),
        cg_window(OTHER_PID, 0, 0, 1000, 1000),
        {"kCGWindowOwnerPID": "junk", "kCGWindowLayer": 0},
    ]
    assert macos._cg_window_at(windows, 150, 150, own_pid=999) == (PID, Rect(50, 50, 800, 600))
    assert macos._cg_window_at(windows, 900, 900, own_pid=999) == (
        OTHER_PID,
        Rect(0, 0, 1000, 1000),
    )
    assert macos._cg_window_at(windows, 5000, 5000, own_pid=999) is None
    assert macos._cg_front_rect(windows, OTHER_PID, own_pid=999) == Rect(0, 0, 1000, 1000)
    assert macos._cg_front_rect(windows, 12, own_pid=999) is None


# ------------------------------------------------------------------ windows
@pytest.mark.usefixtures("appkit")
def test_foreground_window_with_accessibility(
    plat: macos.MacPlatform, ax: FakeAX, quartz: FakeQuartz
) -> None:
    ax.attrs[(f"app:{PID}", "AXFocusedWindow")] = ax.window("win", (100, 50, 800, 600))
    ref = plat.foreground_window()
    assert ref is not None
    assert ref.handle == (PID, "win")
    assert ref.pid == PID
    assert ref.rect == Rect(100, 50, 800, 600)
    # The global AX timeout is set once; the polled app gets a tighter one.
    assert ax.timeouts == [("system", 0.5), (f"app:{PID}", 0.25)]
    plat.foreground_window()
    assert [t for t in ax.timeouts if t[0] == "system"] == [("system", 0.5)]


@pytest.mark.usefixtures("appkit")
def test_foreground_window_without_accessibility(
    plat: macos.MacPlatform, ax: FakeAX, quartz: FakeQuartz
) -> None:
    ax.trusted = False
    quartz.windows = [cg_window(OTHER_PID, 0, 0, 10, 10), cg_window(PID, 1920, 0, 1280, 720)]
    ref = plat.foreground_window()
    assert ref is not None
    assert ref.handle == (PID, None)
    assert ref.rect == Rect(1920, 0, 1280, 720)


def test_foreground_window_skips_our_own_app(
    monkeypatch: pytest.MonkeyPatch, plat: macos.MacPlatform, appkit: FakeAppKit
) -> None:
    monkeypatch.setattr(macos.os, "getpid", lambda: PID)
    assert plat.foreground_window() is None


def test_accessibility_trust_is_cached(plat: macos.MacPlatform, ax: FakeAX, clock: Clock) -> None:
    assert plat._ax_trusted() is True
    ax.trusted = False
    clock.now += 1.0
    assert plat._ax_trusted() is True
    clock.now += 5.0
    assert plat._ax_trusted() is False


def test_window_at_uses_window_list_then_ax(
    plat: macos.MacPlatform, ax: FakeAX, quartz: FakeQuartz
) -> None:
    quartz.windows = [cg_window(PID, 0, 0, 800, 600)]
    back = ax.window("back", (0, 0, 1920, 1080))
    small = ax.window("small", (0, 0, 200, 200))
    ax.window("mini", (0, 0, 800, 600), minimized=True)
    exact = ax.window("exact", (0, 0, 800, 600))
    ax.attrs[(f"app:{PID}", "AXWindows")] = ["mini", small, back, exact]
    ref = plat.window_at(100, 100)
    assert ref is not None
    assert ref.handle == (PID, "exact")  # same frame as the window-list entry
    assert ref.rect == Rect(0, 0, 800, 600)
    ax.attrs[(f"app:{PID}", "AXWindows")] = ["mini", small, back]
    assert plat.window_at(100, 100).handle == (PID, "small")  # type: ignore[union-attr]
    assert plat.window_at(5000, 100) is None


def test_window_at_without_accessibility(
    plat: macos.MacPlatform, ax: FakeAX, quartz: FakeQuartz
) -> None:
    ax.trusted = False
    quartz.windows = [cg_window(PID, 0, 0, 800, 600)]
    ref = plat.window_at(10, 10)
    assert ref is not None
    assert ref.handle == (PID, None)
    assert ref.rect == Rect(0, 0, 800, 600)


def test_window_at_hit_test_fallback(
    monkeypatch: pytest.MonkeyPatch, plat: macos.MacPlatform, ax: FakeAX
) -> None:
    ax.window("win", (0, 0, 640, 480))
    ax.attrs[("text", "AXRole")] = "AXTextArea"
    ax.attrs[("text", "AXWindow")] = "win"
    ax.hit = "text"
    ax.pids["win"] = PID
    ref = plat.window_at(10, 10)  # Quartz missing: AX hit test
    assert ref is not None
    assert ref.handle == (PID, "win")
    assert ref.rect == Rect(0, 0, 640, 480)
    monkeypatch.setattr(macos.os, "getpid", lambda: PID)
    assert plat.window_at(10, 10) is None


@pytest.mark.usefixtures("quartz")
def test_activate_window_full_sequence(
    plat: macos.MacPlatform, ax: FakeAX, appkit: FakeAppKit
) -> None:
    ax.window("win", (0, 0, 10, 10))
    assert plat.activate_window(WindowRef(handle=(PID, "win"), pid=PID)) is True
    assert ("win", "AXMain", True) in ax.sets
    assert (f"app:{PID}", "AXFrontmost", True) in ax.sets
    assert ax.actions == [("win", "AXRaise")]
    assert appkit.apps[PID].activations == [2]


def test_activate_window_refuses_minimised(
    plat: macos.MacPlatform, ax: FakeAX, appkit: FakeAppKit
) -> None:
    ax.window("win", (0, 0, 10, 10), minimized=True)
    assert plat.activate_window(WindowRef(handle=(PID, "win"))) is False
    assert appkit.apps[PID].activations == []
    assert ax.actions == []


def test_activate_app_without_accessibility(
    plat: macos.MacPlatform, ax: FakeAX, appkit: FakeAppKit
) -> None:
    ax.trusted = False
    assert plat.activate_window(WindowRef(handle=(PID, None))) is True
    assert appkit.apps[PID].activations == [2]
    assert ax.sets == []
    appkit.apps[PID].activate_result = False
    assert plat.activate_window(WindowRef(handle=(PID, None))) is False
    assert plat.activate_window(WindowRef(handle=(OTHER_PID, None))) is False  # not running
    assert plat.activate_window(WindowRef(handle="bogus")) is False


def test_is_window_valid(plat: macos.MacPlatform, ax: FakeAX, appkit: FakeAppKit) -> None:
    ax.window("win", (0, 0, 10, 10))
    ax.window("mini", (0, 0, 10, 10), minimized=True)
    assert plat.is_window_valid(WindowRef(handle=(PID, "win")))
    assert plat.is_window_valid(WindowRef(handle=(PID, None)))
    assert not plat.is_window_valid(WindowRef(handle=(PID, "mini")))
    assert not plat.is_window_valid(WindowRef(handle=(PID, "closed")))
    assert not plat.is_window_valid(WindowRef(handle=(OTHER_PID, None)))
    appkit.apps[PID].hidden = True
    assert not plat.is_window_valid(WindowRef(handle=(PID, None)))
    appkit.apps[PID].hidden = False
    appkit.apps[PID].terminated = True
    assert not plat.is_window_valid(WindowRef(handle=(PID, "win")))


def test_window_rect(plat: macos.MacPlatform, ax: FakeAX, quartz: FakeQuartz) -> None:
    ax.window("win", (5, 6, 70, 80))
    quartz.windows = [cg_window(PID, 1, 2, 30, 40)]
    assert plat.window_rect(WindowRef(handle=(PID, "win"))) == Rect(5, 6, 70, 80)
    assert plat.window_rect(WindowRef(handle=(PID, None))) == Rect(1, 2, 30, 40)
    assert plat.window_rect(WindowRef(handle=None)) is None


def test_cursor_position_is_reliable(plat: macos.MacPlatform) -> None:
    assert plat.cursor_position_reliable() is True


# ------------------------------------------------------------------ windows: Spaces
@pytest.mark.usefixtures("appkit")
def test_window_on_another_space_is_not_valid(
    plat: macos.MacPlatform, ax: FakeAX, quartz: FakeQuartz
) -> None:
    # AX keeps answering for a window on another Space (or behind a full-screen
    # app) with a frame inside the display; only the on-screen list tells.
    ax.window("win", (100, 50, 800, 600))
    quartz.windows = [cg_window(OTHER_PID, 0, 0, 1920, 1080)]  # a full-screen app in front
    assert not plat.is_window_valid(WindowRef(handle=(PID, "win")))
    assert not plat.is_window_valid(WindowRef(handle=(PID, None)))
    quartz.windows.append(cg_window(PID, 101, 49, 799, 601))  # back on the visible Space
    assert plat.is_window_valid(WindowRef(handle=(PID, "win")))
    assert plat.is_window_valid(WindowRef(handle=(PID, None)))


@pytest.mark.usefixtures("appkit")
def test_other_window_of_the_same_app_does_not_count(
    plat: macos.MacPlatform, ax: FakeAX, quartz: FakeQuartz
) -> None:
    ax.window("win", (100, 50, 800, 600))
    quartz.windows = [cg_window(PID, 1000, 50, 400, 300)]  # a different window of the app
    assert not plat.is_window_valid(WindowRef(handle=(PID, "win")))
    assert plat.is_window_valid(WindowRef(handle=(PID, None)))


def test_frames_match_tolerance() -> None:
    assert macos._frames_match(Rect(0, 0, 100, 100), Rect(2, -2, 98, 102))
    assert not macos._frames_match(Rect(0, 0, 100, 100), Rect(3, 0, 100, 100))


# ------------------------------------------------------------------ windows: hung apps
@pytest.mark.usefixtures("appkit")
def test_hung_frontmost_app_is_left_alone_for_a_while(
    plat: macos.MacPlatform, ax: FakeAX, quartz: FakeQuartz, clock: Clock
) -> None:
    quartz.windows = [cg_window(PID, 1920, 0, 1280, 720)]
    ax.hung.add(f"app:{PID}")
    ref = plat.foreground_window()
    assert ref is not None
    assert ref.handle == (PID, None)  # application level, from the window list
    assert ref.rect == Rect(1920, 0, 1280, 720)
    asked = len(ax.queries)
    clock.now += 1.0
    again = plat.foreground_window()
    assert again is not None
    assert (again.handle, again.rect) == (ref.handle, ref.rect)
    assert len(ax.queries) == asked  # no further AX request to the hung app
    ax.hung.clear()
    ax.attrs[(f"app:{PID}", "AXFocusedWindow")] = ax.window("win", (1920, 0, 1280, 720))
    clock.now += macos._AX_UNRESPONSIVE_S
    assert plat.foreground_window().handle == (PID, "win")  # type: ignore[union-attr]


def test_ax_rect_stops_after_a_failed_position(ax: FakeAX) -> None:
    ax.window("win", (0, 0, 10, 10))
    ax.hung.add("win")
    assert macos._ax_rect_checked(ax, "win") == (AX_ERROR_CANNOT_COMPLETE, None)
    assert ax.queries == [("win", "AXPosition")]  # AXSize would time out again


@pytest.mark.usefixtures("appkit")
def test_hung_window_is_not_a_target(
    plat: macos.MacPlatform, ax: FakeAX, quartz: FakeQuartz
) -> None:
    ax.window("win", (5, 6, 70, 80))
    quartz.windows = [cg_window(PID, 5, 6, 70, 80)]
    ax.hung.add("win")
    ref = WindowRef(handle=(PID, "win"))
    assert not plat.is_window_valid(ref)
    asked = len(ax.queries)
    assert plat.window_rect(ref) == Rect(5, 6, 70, 80)  # from the window list
    assert not plat.is_window_valid(ref)
    assert len(ax.queries) == asked


def test_activating_a_hung_app_skips_accessibility(
    plat: macos.MacPlatform, ax: FakeAX, appkit: FakeAppKit
) -> None:
    ax.window("win", (0, 0, 10, 10))
    ax.hung.add("win")
    assert plat.activate_window(WindowRef(handle=(PID, "win"))) is True
    assert appkit.apps[PID].activations == [2]
    assert ax.sets == []
    assert ax.actions == []
    assert ax.queries == [("win", "AXMinimized")]


def test_same_window(plat: macos.MacPlatform) -> None:
    compared: list[tuple[Any, Any]] = []

    def cf_equal(a: Any, b: Any) -> bool:
        compared.append((a, b))
        return a == b

    plat._modules["CoreFoundation"] = SimpleNamespace(CFEqual=cf_equal)
    a = WindowRef(handle=(PID, "w1"))
    assert plat.same_window(a, WindowRef(handle=(PID, "w1")))
    assert not plat.same_window(a, WindowRef(handle=(PID, "w2")))
    assert not plat.same_window(a, WindowRef(handle=(OTHER_PID, "w1")))
    assert not plat.same_window(a, WindowRef(handle=(PID, None)))
    assert plat.same_window(WindowRef(handle=(PID, None)), WindowRef(handle=(PID, None)))
    assert not plat.same_window(a, None)
    assert compared == [("w1", "w1"), ("w1", "w2")]


# ------------------------------------------------------------------ permissions
def test_accessibility_permission_and_prompt(plat: macos.MacPlatform, ax: FakeAX) -> None:
    assert plat.permissions()["accessibility"] is True
    ax.trusted = False
    assert plat.permissions()["accessibility"] is False
    plat.request_permission("accessibility")
    assert ax.prompts == [{"AXTrustedCheckOptionPrompt": True}]
    ax.kAXTrustedCheckOptionPrompt = "kPrompt"  # type: ignore[attr-defined]
    plat.request_permission("accessibility")
    assert ax.prompts[-1] == {"kPrompt": True}


@pytest.mark.parametrize(("status", "expected"), [(0, None), (1, False), (2, False), (3, True)])
def test_camera_permission_via_avfoundation(
    plat: macos.MacPlatform, status: int, expected: bool | None
) -> None:
    queried: list[str] = []

    def auth(media_type: str) -> int:
        queried.append(media_type)
        return status

    device = SimpleNamespace(authorizationStatusForMediaType_=auth)
    plat._modules["AVFoundation"] = SimpleNamespace(AVCaptureDevice=device, AVMediaTypeVideo="vide")
    assert plat.permissions()["camera"] is expected
    assert queried == ["vide"]


def test_camera_permission_via_objc_runtime(plat: macos.MacPlatform) -> None:
    bundles: list[tuple[str, dict[str, Any]]] = []

    def load_bundle(name: str, module_globals: dict[str, Any], **kwargs: Any) -> None:
        bundles.append((name, kwargs))

    device = SimpleNamespace(
        authorizationStatusForMediaType_=lambda media: 3 if media == "vide" else 0
    )
    plat._modules["objc"] = SimpleNamespace(
        loadBundle=load_bundle,
        lookUpClass=lambda name: device if name == "AVCaptureDevice" else None,
    )
    assert plat.permissions()["camera"] is True
    assert bundles == [
        ("AVFoundation", {"bundle_path": macos.AVFOUNDATION_FRAMEWORK, "scan_classes": False})
    ]


@pytest.mark.usefixtures("on_macos")
def test_open_permission_settings(plat: macos.MacPlatform, recorder: Recorder) -> None:
    recorder.tools = {"open": "/usr/bin/open"}
    assert plat.open_permission_settings("accessibility") is True
    assert plat.open_permission_settings("camera") is True
    assert plat.open_permission_settings("bluetooth") is False
    assert recorder.runs == [
        ["/usr/bin/open", macos._PERMISSION_URLS["accessibility"]],
        ["/usr/bin/open", macos._PERMISSION_URLS["camera"]],
    ]
    assert recorder.runs[0][1].endswith("Privacy_Accessibility")


def test_set_accessory_app(plat: macos.MacPlatform, appkit: FakeAppKit) -> None:
    assert plat.set_accessory_app() is True
    assert appkit.policies == [1]


# ------------------------------------------------------------------ App Nap
class FakeProcessInfo:
    def __init__(self) -> None:
        self.begun: list[tuple[int, str]] = []
        self.ended: list[Any] = []
        self.fail = False

    def beginActivityWithOptions_reason_(self, options: int, reason: str) -> Any:
        if self.fail:
            raise RuntimeError("no activities here")
        self.begun.append((options, reason))
        return f"token{len(self.begun)}"

    def endActivity_(self, token: Any) -> None:
        self.ended.append(token)


@pytest.fixture
def process_info(plat: macos.MacPlatform) -> FakeProcessInfo:
    info = FakeProcessInfo()
    plat._modules["Foundation"] = SimpleNamespace(
        NSProcessInfo=SimpleNamespace(processInfo=lambda: info)
    )
    return info


@pytest.mark.usefixtures("appkit")
def test_accessory_app_opts_out_of_app_nap(
    plat: macos.MacPlatform, process_info: FakeProcessInfo
) -> None:
    assert plat.set_accessory_app() is True
    # User initiated (no App Nap, no timer coalescing) but idle sleep still allowed.
    assert process_info.begun == [(0x00EFFFFF, macos._ACTIVITY_REASON)]
    assert plat._activity == "token1"  # the token must stay referenced


def test_background_activity_begins_and_ends_once(
    plat: macos.MacPlatform, process_info: FakeProcessInfo
) -> None:
    assert plat.set_background_activity(True) is True
    assert plat.set_background_activity(True) is True
    assert len(process_info.begun) == 1
    assert plat.set_background_activity(False) is True
    assert plat.set_background_activity(False) is True
    assert process_info.ended == ["token1"]
    assert plat._activity is None
    assert plat.set_background_activity(True) is True  # e.g. tracking resumed
    assert plat._activity == "token2"


def test_background_activity_uses_the_named_option(plat: macos.MacPlatform) -> None:
    info = FakeProcessInfo()
    plat._modules["Foundation"] = SimpleNamespace(
        NSProcessInfo=SimpleNamespace(processInfo=lambda: info),
        NSActivityUserInitiatedAllowingIdleSystemSleep=0x1234,
    )
    assert plat.set_background_activity(True) is True
    assert info.begun[0][0] == 0x1234


def test_background_activity_degrades(
    plat: macos.MacPlatform, process_info: FakeProcessInfo
) -> None:
    process_info.fail = True
    assert plat.set_background_activity(True) is False
    assert plat._activity is None
    plat._modules["Foundation"] = None
    assert plat.set_background_activity(True) is False
    assert plat.set_background_activity(False) is True  # nothing to end


# ------------------------------------------------------------------ Accessibility after updates
def test_accessibility_status_granted_and_missing(
    plat: macos.MacPlatform, ax: FakeAX, state_path: Path
) -> None:
    assert plat.accessibility_status() == "granted"
    ax.trusted = False
    assert plat.accessibility_status() == "missing"
    assert not state_path.exists()  # source runs have no build identity to remember


def test_accessibility_status_unknown_without_the_api(plat: macos.MacPlatform) -> None:
    assert plat.accessibility_status() == "unknown"


def _packaged(monkeypatch: pytest.MonkeyPatch, build: str) -> None:
    monkeypatch.setattr(macos, "_build_fingerprint", lambda: build)


def test_accessibility_granted_to_an_earlier_build_is_stale(
    monkeypatch: pytest.MonkeyPatch, clock: Clock, state_path: Path
) -> None:
    fake = FakeAX()
    _packaged(monkeypatch, "1.0:100:1")
    first = macos.MacPlatform(clock=clock, state_path=state_path)
    first._modules["ApplicationServices"] = fake
    assert first.accessibility_status() == "granted"
    assert json.loads(state_path.read_text(encoding="utf-8")) == {"trusted_build": "1.0:100:1"}

    # An update: new ad-hoc signature, System Settings still shows the old grant.
    fake.trusted = False
    _packaged(monkeypatch, "1.1:120:2")
    second = macos.MacPlatform(clock=clock, state_path=state_path)
    second._modules["ApplicationServices"] = fake
    assert second.accessibility_status() == "stale"
    assert second.permissions()["accessibility"] is False

    # Re-granted: the record follows the new build.
    fake.trusted = True
    assert second.accessibility_status() == "granted"
    assert json.loads(state_path.read_text(encoding="utf-8")) == {"trusted_build": "1.1:120:2"}


def test_accessibility_never_granted_is_missing_for_packaged_builds(
    monkeypatch: pytest.MonkeyPatch, plat: macos.MacPlatform, ax: FakeAX, state_path: Path
) -> None:
    _packaged(monkeypatch, "1.0:100:1")
    ax.trusted = False
    assert plat.accessibility_status() == "missing"
    state_path.write_text("not json", encoding="utf-8")  # a damaged record
    fresh = macos.MacPlatform(state_path=state_path)
    fresh._modules["ApplicationServices"] = ax
    assert fresh.accessibility_status() == "missing"


def _app_bundle(root: Path, *, plist: bytes | None = None) -> tuple[Path, Path]:
    """``Eye Tracker.app`` as the spec builds it: ``(app executable, eye-tracker-cli)``."""
    macos_dir = root / "Eye Tracker.app" / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True)
    app = macos_dir / "Eye Tracker"
    cli = macos_dir / "eye-tracker-cli"
    app.write_bytes(b"\0" * 300)  # written and signed separately: sizes and mtimes differ
    cli.write_bytes(b"\0" * 200)
    if plist is None:
        plist = plistlib.dumps({"CFBundleExecutable": "Eye Tracker"})
    (macos_dir.parent / "Info.plist").write_bytes(plist)
    return app, cli


def _run_as(monkeypatch: pytest.MonkeyPatch, executable: Path) -> None:
    monkeypatch.setattr(macos.sys, "frozen", True, raising=False)
    monkeypatch.setattr(macos.sys, "executable", str(executable))


def test_build_fingerprint_is_the_apps_in_both_executables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, cli = _app_bundle(tmp_path)
    _run_as(monkeypatch, app)
    from_app = macos._build_fingerprint()
    assert from_app is not None
    assert from_app.endswith(":300:" + str(app.stat().st_mtime_ns))
    assert macos._runs_helper_executable() is False
    _run_as(monkeypatch, cli)
    assert macos._build_fingerprint() == from_app
    assert macos._runs_helper_executable() is True


@pytest.mark.parametrize(
    "plist",
    [b"not a plist", plistlib.dumps({"CFBundleExecutable": "../evil"}), plistlib.dumps([])],
)
def test_build_fingerprint_without_a_usable_info_plist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plist: bytes
) -> None:
    _app, cli = _app_bundle(tmp_path, plist=plist)
    _run_as(monkeypatch, cli)
    fingerprint = macos._build_fingerprint()
    assert fingerprint is not None
    assert ":200:" in fingerprint  # its own file, as before
    assert macos._runs_helper_executable() is True  # still known by its name


def test_cli_does_not_judge_or_record_the_apps_accessibility_grant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock, state_path: Path
) -> None:
    """`eye-tracker-cli doctor` from Terminal: macOS asks about Terminal, not the app."""
    app, cli = _app_bundle(tmp_path)
    _run_as(monkeypatch, app)
    gui = macos.MacPlatform(clock=clock, state_path=state_path)
    fake = FakeAX()
    gui._modules["ApplicationServices"] = fake
    assert gui.accessibility_status() == "granted"
    recorded = state_path.read_text(encoding="utf-8")

    _run_as(monkeypatch, cli)
    fake.trusted = False  # Terminal has no Accessibility access
    doctor = macos.MacPlatform(clock=clock, state_path=state_path)
    doctor._modules["ApplicationServices"] = fake
    assert doctor.accessibility_status() == "unknown"  # not "stale", not "missing"
    assert doctor.permissions()["accessibility"] is None
    assert state_path.read_text(encoding="utf-8") == recorded

    # Terminal has Accessibility access: not a grant of this (or any) app build.
    state_path.write_text(json.dumps({"trusted_build": "0.9:1:1"}), encoding="utf-8")
    fake.trusted = True
    trusted_terminal = macos.MacPlatform(clock=clock, state_path=state_path)
    trusted_terminal._modules["ApplicationServices"] = fake
    assert trusted_terminal.accessibility_status() == "granted"
    assert trusted_terminal.permissions()["accessibility"] is True
    assert json.loads(state_path.read_text(encoding="utf-8")) == {"trusted_build": "0.9:1:1"}


def test_activation_without_accessibility_fails_on_sonoma(
    monkeypatch: pytest.MonkeyPatch,
    plat: macos.MacPlatform,
    ax: FakeAX,
    appkit: FakeAppKit,
    state_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # macOS 14+ ignores activation requests from an inactive (menu-bar) app.
    monkeypatch.setattr(macos, "_macos_major_version", lambda: 14)
    _packaged(monkeypatch, "1.1:120:2")
    state_path.write_text(json.dumps({"trusted_build": "1.0:100:1"}), encoding="utf-8")
    ax.trusted = False
    ref = WindowRef(handle=(PID, None))
    with caplog.at_level(logging.WARNING, logger=macos.__name__):
        assert plat.activate_window(ref) is False
        assert plat.activate_window(ref) is False
    assert appkit.apps[PID].activations == [2, 2]  # still asked, in case it works
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "earlier version" in warnings[0]
    monkeypatch.setattr(macos, "_macos_major_version", lambda: 13)
    assert plat.activate_window(ref) is True


@pytest.mark.usefixtures("on_macos", "quartz", "appkit", "ax")
def test_capabilities_with_frameworks(plat: macos.MacPlatform, recorder: Recorder) -> None:
    recorder.tools = {"pmset": "/usr/bin/pmset", "caffeinate": "/usr/bin/caffeinate"}
    caps = plat.capabilities()
    assert set(caps) == set(PlatformServices().capabilities())
    assert all(caps[key] for key in ("lock", "display_off", "wake_display", "input_idle"))
    assert all(caps[key] for key in ("key_idle", "session_locked", "focus", "cursor", "hotkeys"))
    assert caps["camera_in_use"] is False


def test_module_contract() -> None:
    assert macos.MacPlatform.name == "macos"
    assert issubclass(macos.MacPlatform, PlatformServices)
