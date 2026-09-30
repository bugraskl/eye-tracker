"""Tests for global hotkey parsing, formatting and the native managers.

Nothing here synthesises key presses. On Windows the live tests register an
unusual combination and deliver ``WM_HOTKEY`` to the manager's own thread with
``PostThreadMessageW``; macOS and X11 code paths run against fakes (plus an
isolated live smoke test on those systems).
"""

from __future__ import annotations

import collections
import logging
import os
import subprocess
import sys
import threading
import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from eye_tracker.platform import hotkeys
from eye_tracker.platform.hotkeys import (
    KEYS,
    MODIFIERS,
    Hotkey,
    HotkeyManager,
    MacHotkeyManager,
    WindowsHotkeyManager,
    X11HotkeyManager,
    create_hotkey_manager,
    format_hotkey,
    parse_hotkey,
)

# Unusual combinations that no desktop or application binds by default.
COMBO = "ctrl+alt+shift+f24"
COMBO_2 = "ctrl+alt+shift+f21"
MAC_COMBO = "ctrl+alt+shift+f20"  # macOS keyboards stop at F20

IS_WINDOWS = sys.platform == "win32"
IS_POSIX = os.name == "posix"


def hk(text: str) -> Hotkey:
    return parse_hotkey(text)


# =========================================================================== parsing
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ctrl+Alt+P", "ctrl+alt+p"),
        ("ctrl+alt+p", "ctrl+alt+p"),
        ("CTRL+ALT+P", "ctrl+alt+p"),
        ("  ctrl +  alt+ p  ", "ctrl+alt+p"),
        ("alt+ctrl+p", "ctrl+alt+p"),
        ("Ctrl+Shift+F5", "ctrl+shift+f5"),
        ("ctrl+alt+shift+meta+f24", "ctrl+alt+shift+meta+f24"),
        ("Ctrl+Alt+Space", "ctrl+alt+space"),
        ("ctrl+alt+esc", "ctrl+alt+escape"),
        ("ctrl+alt+Return", "ctrl+alt+enter"),
        ("ctrl+alt+Del", "ctrl+alt+delete"),
        ("ctrl+alt+Ins", "ctrl+alt+insert"),
        ("ctrl+alt+PgUp", "ctrl+alt+pageup"),
        ("ctrl+alt+PgDown", "ctrl+alt+pagedown"),
        ("ctrl+alt+Page Up", "ctrl+alt+pageup"),
        ("ctrl+alt+page_down", "ctrl+alt+pagedown"),
        ("ctrl+alt+Left", "ctrl+alt+left"),
        ("ctrl+alt+Backspace", "ctrl+alt+backspace"),
        ("ctrl+alt+Tab", "ctrl+alt+tab"),
        ("ctrl+alt+Home", "ctrl+alt+home"),
        ("ctrl+alt+End", "ctrl+alt+end"),
        ("ctrl+alt+7", "ctrl+alt+7"),
        ("ctrl+alt+-", "ctrl+alt+minus"),
        ("ctrl+alt+=", "ctrl+alt+equal"),
        ("ctrl+alt+[", "ctrl+alt+bracketleft"),
        ("ctrl+alt+]", "ctrl+alt+bracketright"),
        ("ctrl+alt+\\", "ctrl+alt+backslash"),
        ("ctrl+alt+;", "ctrl+alt+semicolon"),
        ("ctrl+alt+'", "ctrl+alt+quote"),
        ("ctrl+alt+`", "ctrl+alt+grave"),
        ("ctrl+alt+,", "ctrl+alt+comma"),
        ("ctrl+alt+.", "ctrl+alt+period"),
        ("ctrl+alt+/", "ctrl+alt+slash"),
        ("ctrl+alt+comma", "ctrl+alt+comma"),
        ("shift+f5", "shift+f5"),  # Shift alone is fine for non-typing keys
        ("shift+pageup", "shift+pageup"),
        ("⌃⌥P", "ctrl+alt+p"),
        ("⌃⌥⇧⌘F24", "ctrl+alt+shift+meta+f24"),
        ("⌘⇧G", "shift+meta+g"),
        ("⌃ + ⌥ + P", "ctrl+alt+p"),
        ("ctrl+⌥P", "ctrl+alt+p"),
        ("⌘Space", "meta+space"),
        ("⌃⌥⎋", "ctrl+alt+escape"),
        ("⌃⌥←", "ctrl+alt+left"),
    ],
)
def test_parse_valid(text: str, expected: str) -> None:
    assert str(parse_hotkey(text)) == expected


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("cmd", "meta"),
        ("Command", "meta"),
        ("win", "meta"),
        ("Windows", "meta"),
        ("super", "meta"),
        ("META", "meta"),
        ("option", "alt"),
        ("Opt", "alt"),
        ("control", "ctrl"),
        ("Ctl", "ctrl"),
        ("Shift", "shift"),
    ],
)
def test_modifier_aliases(alias: str, canonical: str) -> None:
    other = "shift" if canonical != "shift" else "ctrl"
    parsed = parse_hotkey(f"{alias}+{other}+g")
    assert parsed.modifiers == frozenset({canonical, other})
    assert parsed.key == "g"


def test_parse_is_case_insensitive_and_whitespace_tolerant() -> None:
    variants = ["Ctrl+Alt+P", "ctrl+alt+p", "cTrL+aLt+P", " Ctrl + Alt + P ", "\tctrl+alt+p\n"]
    assert {parse_hotkey(v) for v in variants} == {Hotkey(frozenset({"ctrl", "alt"}), "p")}


def test_duplicate_modifiers_are_merged() -> None:
    assert parse_hotkey("ctrl+control+p") == parse_hotkey("ctrl+p")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "p",  # key only
        "F5",  # key only
        "space",
        "ctrl+alt",  # modifiers only
        "ctrl",
        "⌃⌥",
        "ctrl+",  # missing key
        "+p",  # empty modifier
        "ctrl++p",
        "ctrl++",  # the '+' key
        "+",
        "hyper+p",  # unknown modifier
        "ctrl+alt+p+q",  # 'p' is not a modifier
        "ctrl+f25",
        "ctrl+f0",
        "ctrl+alt+pp",  # unknown key
        "ctrl+alt+é",
        "shift+a",  # would block typing capitals
        "shift+1",
        "shift+space",
        "shift+tab",
        "shift+enter",
        "shift+/",
    ],
)
def test_parse_invalid_raises(text: str) -> None:
    with pytest.raises(ValueError, match=r"\S"):
        parse_hotkey(text)


def test_parse_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="string"):
        parse_hotkey(None)  # type: ignore[arg-type]


def test_hotkey_validation_and_normalisation() -> None:
    built = Hotkey(["alt", "ctrl"], "p")  # type: ignore[arg-type]
    assert built.modifiers == frozenset({"ctrl", "alt"})
    assert built == parse_hotkey("ctrl+alt+p")
    assert hash(built) == hash(parse_hotkey("alt+ctrl+p"))
    assert built.ordered_modifiers == ("ctrl", "alt")
    with pytest.raises(ValueError, match="modifier"):
        Hotkey(frozenset({"hyper"}), "p")
    with pytest.raises(ValueError, match="key"):
        Hotkey(frozenset({"ctrl"}), "P")  # keys are canonical lower case
    with pytest.raises(ValueError, match="at least one modifier"):
        Hotkey(frozenset(), "p")
    with pytest.raises(ValueError, match="Shift alone"):
        Hotkey(frozenset({"shift"}), "a")


def test_str_is_canonical_and_round_trips() -> None:
    parsed = parse_hotkey("Win+Shift+Alt+Ctrl+K")
    assert str(parsed) == "ctrl+alt+shift+meta+k"
    assert parse_hotkey(str(parsed)) == parsed


def test_default_settings_hotkeys_parse() -> None:
    from eye_tracker.config import HotkeySettings

    defaults = HotkeySettings()
    for text in (defaults.toggle_tracking, defaults.toggle_privacy, defaults.recalibrate):
        assert parse_hotkey(text).modifiers == frozenset({"ctrl", "alt"})


def test_key_table_is_complete() -> None:
    assert MODIFIERS == ("ctrl", "alt", "shift", "meta")
    assert len(KEYS) == len(set(KEYS))
    for key in ("a", "z", "0", "9", "f1", "f24", "space", "escape", "tab", "enter"):
        assert key in KEYS
    for key in ("left", "right", "up", "down", "home", "end", "pageup", "pagedown"):
        assert key in KEYS
    for key in ("insert", "delete", "minus", "equal", "comma", "period", "slash"):
        assert key in KEYS


# ========================================================================= formatting
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ctrl+alt+p", "Ctrl+Alt+P"),
        ("shift+ctrl+f5", "Ctrl+Shift+F5"),
        ("ctrl+alt+space", "Ctrl+Alt+Space"),
        ("ctrl+alt+esc", "Ctrl+Alt+Esc"),
        ("ctrl+alt+pageup", "Ctrl+Alt+PgUp"),
        ("ctrl+alt+-", "Ctrl+Alt+-"),
    ],
)
def test_format_default_style(text: str, expected: str) -> None:
    assert format_hotkey(parse_hotkey(text), macos=False) == expected


def test_format_meta_label_follows_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    hotkey = parse_hotkey("win+alt+g")
    monkeypatch.setattr(sys, "platform", "win32")
    assert format_hotkey(hotkey, macos=False) == "Alt+Win+G"
    monkeypatch.setattr(sys, "platform", "linux")
    assert format_hotkey(hotkey, macos=False) == "Alt+Super+G"
    assert format_hotkey(hotkey) == "Alt+Super+G"  # macos=None follows sys.platform
    monkeypatch.setattr(sys, "platform", "darwin")
    assert format_hotkey(hotkey) == "⌥⌘G"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ctrl+alt+p", "⌃⌥P"),
        ("cmd+shift+g", "⇧⌘G"),
        ("ctrl+alt+shift+cmd+f12", "⌃⌥⇧⌘F12"),
        ("cmd+space", "⌘Space"),
        ("ctrl+alt+esc", "⌃⌥⎋"),
        ("ctrl+alt+enter", "⌃⌥↩"),
        ("ctrl+alt+left", "⌃⌥←"),
    ],
)
def test_format_macos_style(text: str, expected: str) -> None:
    assert format_hotkey(parse_hotkey(text), macos=True) == expected


@pytest.mark.parametrize("macos", [False, True])
@pytest.mark.parametrize(
    "modifiers",
    [{"ctrl", "alt"}, {"meta"}, {"ctrl", "alt", "shift", "meta"}, {"alt", "shift"}],
)
def test_format_parse_round_trip_for_every_key(macos: bool, modifiers: set[str]) -> None:
    for key in KEYS:
        hotkey = Hotkey(frozenset(modifiers), key)
        assert parse_hotkey(format_hotkey(hotkey, macos=macos)) == hotkey, key


# =============================================================== unsupported manager
def test_base_manager_is_a_safe_no_op() -> None:
    manager = HotkeyManager(note="nope")
    assert manager.supported is False
    assert manager.note == "nope"
    manager.start()
    assert manager.register("t", parse_hotkey(COMBO), lambda: None) is False
    assert manager.unregister("t") is False
    assert manager.registered == {}
    manager.unregister_all()
    manager.stop()
    manager.stop()


# ============================================================================ factory
def test_factory_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    manager = create_hotkey_manager()  # constructing starts nothing
    assert isinstance(manager, WindowsHotkeyManager)
    assert manager.supported


def test_factory_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    manager = create_hotkey_manager()  # Carbon is loaded lazily
    assert isinstance(manager, MacHotkeyManager)


def test_factory_wayland_without_display_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    manager = create_hotkey_manager()
    assert type(manager) is HotkeyManager
    assert not manager.supported
    assert manager.note is not None
    assert "ctl" in manager.note


def test_factory_headless_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    manager = create_hotkey_manager()
    assert type(manager) is HotkeyManager


def test_factory_x11_without_xlib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setitem(sys.modules, "Xlib", None)  # makes `import Xlib` fail
    manager = create_hotkey_manager()
    assert type(manager) is HotkeyManager
    assert manager.note is not None
    assert "python-xlib" in manager.note


@pytest.mark.parametrize("wayland", [False, True])
def test_factory_x11(monkeypatch: pytest.MonkeyPatch, wayland: bool) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland" if wayland else "x11")
    monkeypatch.setitem(sys.modules, "Xlib", types.ModuleType("Xlib"))
    manager = create_hotkey_manager()  # does not connect to the X server yet
    assert isinstance(manager, X11HotkeyManager)
    assert (manager.note is not None) is wayland


# ============================================================ Windows virtual keys
def test_windows_virtual_key_table() -> None:
    for key in KEYS:
        assert hotkeys._win_virtual_key(key) is not None, key
    assert hotkeys._win_virtual_key("a") == 0x41
    assert hotkeys._win_virtual_key("z") == 0x5A
    assert hotkeys._win_virtual_key("0") == 0x30
    assert hotkeys._win_virtual_key("f1") == 0x70
    assert hotkeys._win_virtual_key("f24") == 0x87
    assert hotkeys._win_virtual_key("escape") == 0x1B
    assert hotkeys._win_virtual_key("slash") == 0xBF


def test_windows_virtual_key_prefers_active_layout_for_punctuation() -> None:
    # Unshifted mapping from the layout wins …
    assert hotkeys._win_virtual_key("period", lambda ch: 0xBF) == 0xBF
    # … a mapping that needs Shift/AltGr, or none at all, falls back to the US table.
    assert hotkeys._win_virtual_key("period", lambda ch: 0x0137) == 0xBE
    assert hotkeys._win_virtual_key("period", lambda ch: -1) == 0xBE
    # Letters never consult the layout.
    assert hotkeys._win_virtual_key("q", lambda ch: 0x99) == ord("Q")


def test_windows_modifier_mask() -> None:
    assert hotkeys._win_modifiers({"ctrl", "alt"}) == 0x0003
    assert hotkeys._win_modifiers({"shift", "meta"}) == 0x000C


# ======================================================== generic threaded machinery
class LoopbackManager(hotkeys._ThreadedHotkeyManager):
    """A portable threaded manager whose "OS" is a dict, to test the shared plumbing."""

    name = "loopback"

    def __init__(self) -> None:
        super().__init__()
        self._wake_event = threading.Event()
        self.os_registered: dict[int, Hotkey] = {}
        self.taken: set[Hotkey] = set()
        self.native_threads: set[str] = set()
        self.block: threading.Event | None = None

    def _thread_setup(self) -> bool:
        return True

    def _thread_loop(self) -> None:
        while not self._stopping.is_set():
            self._wake_event.wait(0.2)
            self._wake_event.clear()
            self._run_pending_calls()

    def _thread_teardown(self) -> None:
        self.os_registered.clear()

    def _wake(self) -> bool:
        self._wake_event.set()
        return True

    def _native_register(self, binding: Any) -> bool:
        return bool(self._call_on_thread(lambda: self._reg(binding), False))

    def _native_unregister(self, binding: Any) -> None:
        self._call_on_thread(lambda: self.os_registered.pop(binding.id, None), None)

    def _reg(self, binding: Any) -> bool:
        self.native_threads.add(threading.current_thread().name)
        if self.block is not None:
            self.block.wait(1.0)
        if binding.hotkey in self.taken:
            return False
        self.os_registered[binding.id] = binding.hotkey
        return True

    def fire(self, name: str) -> None:
        binding_id = self._bindings[name].id
        self._call_on_thread(lambda: self._dispatch(binding_id), None)


@pytest.fixture
def loopback() -> Iterator[LoopbackManager]:
    manager = LoopbackManager()
    try:
        yield manager
    finally:
        manager.stop()


def test_threaded_registration_runs_on_backend_thread(loopback: LoopbackManager) -> None:
    calls: list[str] = []
    assert loopback.register("t", COMBO, lambda: calls.append(threading.current_thread().name))
    assert loopback.native_threads == {"eye-tracker-hotkeys-loopback"}
    assert loopback.registered == {"t": parse_hotkey(COMBO)}
    loopback.fire("t")
    assert calls == ["eye-tracker-hotkeys-loopback"]


def test_double_registration_semantics(loopback: LoopbackManager) -> None:
    fired: list[str] = []
    assert loopback.register("a", COMBO, lambda: fired.append("first"))
    # Same name, same combo: the callback is replaced without touching the OS.
    assert loopback.register("a", COMBO, lambda: fired.append("second"))
    assert len(loopback.os_registered) == 1
    loopback.fire("a")
    assert fired == ["second"]
    # Another name for a combo this manager already owns is refused.
    assert not loopback.register("b", COMBO, lambda: None)
    assert set(loopback.registered) == {"a"}
    # Re-binding a name to a new combo releases the old one.
    assert loopback.register("a", COMBO_2, lambda: None)
    assert list(loopback.os_registered.values()) == [parse_hotkey(COMBO_2)]
    assert loopback.register("b", COMBO, lambda: None)


def test_os_rejection_leaves_no_trace(loopback: LoopbackManager) -> None:
    loopback.taken.add(parse_hotkey(COMBO))
    assert not loopback.register("t", COMBO, lambda: None)
    assert loopback.registered == {}
    assert loopback._by_id == {}


def test_invalid_register_arguments(loopback: LoopbackManager) -> None:
    assert not loopback.register("t", "not a hotkey", lambda: None)
    with pytest.raises(TypeError):
        loopback.register("t", COMBO, "not callable")  # type: ignore[arg-type]


def test_unregister_and_unregister_all(loopback: LoopbackManager) -> None:
    assert loopback.register("a", COMBO, lambda: None)
    assert loopback.register("b", COMBO_2, lambda: None)
    assert loopback.unregister("a")
    assert not loopback.unregister("a")
    assert set(loopback.registered) == {"b"}
    loopback.unregister_all()
    assert loopback.registered == {}
    assert loopback.os_registered == {}
    thread = loopback._thread
    assert thread is not None
    assert thread.is_alive()  # unregister_all keeps the backend running


def test_stop_is_idempotent_and_restartable(loopback: LoopbackManager) -> None:
    loopback.stop()  # before anything started
    loopback.start()
    thread = loopback._thread
    assert thread is not None
    assert thread.is_alive()
    assert loopback.register("t", COMBO, lambda: None)
    loopback.stop()
    assert not thread.is_alive()
    assert loopback.registered == {}
    loopback.stop()
    assert loopback.register("t", COMBO, lambda: None)  # lazily restarts
    new_thread = loopback._thread
    assert new_thread is not None
    assert new_thread is not thread
    assert new_thread.is_alive()


def test_callback_exception_is_logged(
    loopback: LoopbackManager, caplog: pytest.LogCaptureFixture
) -> None:
    def boom() -> None:
        raise RuntimeError("callback failure")

    fired: list[int] = []
    assert loopback.register("bad", COMBO, boom)
    assert loopback.register("good", COMBO_2, lambda: fired.append(1))
    with caplog.at_level(logging.ERROR, logger="eye_tracker.platform.hotkeys"):
        loopback.fire("bad")
    assert "callback failure" in caplog.text
    loopback.fire("good")
    assert fired == [1]


def test_call_timeout_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = LoopbackManager()
    monkeypatch.setattr(manager, "_CALL_TIMEOUT_S", 0.05)
    manager.start()
    try:
        release = threading.Event()
        # Occupy the backend thread so the next call cannot start in time.
        manager._calls.put(hotkeys._PendingCall(lambda: release.wait(1.0)))
        manager._wake()
        assert manager._call_on_thread(lambda: "late", "default") == "default"
        release.set()
    finally:
        manager.stop()


def test_dispatch_ignores_unknown_ids(loopback: LoopbackManager) -> None:
    loopback._dispatch(12345)  # must not raise


# ============================================================================== macOS
class FakeCarbon:
    """Stand-in for the ctypes Carbon binding."""

    def __init__(self) -> None:
        self.handler: Callable[[Any], int] | None = None
        self.install_calls = 0
        self.removed: list[Any] = []
        self.registered: dict[str, tuple[int, int, int, int]] = {}
        self.reject: dict[tuple[int, int], int] = {}
        self._next = 0

    def install_handler(self, handler: Callable[[Any], int]) -> tuple[int, Any]:
        self.install_calls += 1
        self.handler = handler
        return 0, "handler-ref"

    def remove_handler(self, ref: Any) -> int:
        self.removed.append(ref)
        self.handler = None
        return 0

    def register_hotkey(
        self, keycode: int, modifiers: int, signature: int, hotkey_id: int
    ) -> tuple[int, Any]:
        status = self.reject.get((keycode, modifiers), 0)
        if status:
            return status, None
        self._next += 1
        ref = f"ref{self._next}"
        self.registered[ref] = (keycode, modifiers, signature, hotkey_id)
        return 0, ref

    def unregister_hotkey(self, ref: Any) -> int:
        self.registered.pop(ref)
        return 0

    def hotkey_id(self, event: Any) -> tuple[int, int, int]:
        return event  # the fake "EventRef" already is (status, signature, id)

    def press(self, keycode: int, modifiers: int) -> int:
        assert self.handler is not None
        for code, mods, signature, hotkey_id in self.registered.values():
            if (code, mods) == (keycode, modifiers):
                return self.handler((0, signature, hotkey_id))
        raise AssertionError("combination not registered")


@pytest.fixture
def carbon() -> FakeCarbon:
    return FakeCarbon()


@pytest.fixture
def mac(carbon: FakeCarbon) -> Iterator[MacHotkeyManager]:
    manager = MacHotkeyManager(carbon=carbon)
    try:
        yield manager
    finally:
        manager.stop()


def test_mac_register_uses_carbon_codes(mac: MacHotkeyManager, carbon: FakeCarbon) -> None:
    fired: list[str] = []
    assert mac.register("t", "ctrl+alt+p", lambda: fired.append("t"))
    assert carbon.install_calls == 1
    ((keycode, modifiers, signature, hotkey_id),) = carbon.registered.values()
    assert keycode == 0x23  # kVK_ANSI_P
    assert modifiers == 0x1000 | 0x0800  # controlKey | optionKey
    assert signature == hotkeys._fourcc("EyTk")
    assert hotkey_id == mac._bindings["t"].id
    assert carbon.press(0x23, 0x1800) == 0  # noErr
    assert fired == ["t"]


def test_mac_modifier_masks() -> None:
    assert hotkeys._mac_modifiers({"meta"}) == 0x0100
    assert hotkeys._mac_modifiers({"shift"}) == 0x0200
    assert hotkeys._mac_modifiers({"alt"}) == 0x0800
    assert hotkeys._mac_modifiers({"ctrl"}) == 0x1000


def test_mac_foreign_events_are_passed_on(mac: MacHotkeyManager, carbon: FakeCarbon) -> None:
    assert mac.register("t", "cmd+shift+g", lambda: None)
    assert carbon.handler is not None
    assert carbon.handler((0, hotkeys._fourcc("XXXX"), 1)) == -9874  # eventNotHandledErr
    assert carbon.handler((-50, 0, 0)) == -9874


def test_mac_double_registration_and_rebinding(mac: MacHotkeyManager, carbon: FakeCarbon) -> None:
    assert mac.register("a", MAC_COMBO, lambda: None)
    assert mac.register("a", MAC_COMBO, lambda: None)
    assert len(carbon.registered) == 1
    assert not mac.register("b", MAC_COMBO, lambda: None)
    assert mac.register("a", "ctrl+alt+p", lambda: None)
    assert [v[0] for v in carbon.registered.values()] == [0x23]


def test_mac_rejections(mac: MacHotkeyManager, carbon: FakeCarbon) -> None:
    carbon.reject[(0x23, 0x1800)] = -9878  # eventHotKeyExistsErr
    assert not mac.register("t", "ctrl+alt+p", lambda: None)
    assert not mac.register("t", "ctrl+alt+f24", lambda: None)  # no F24 on a Mac
    assert mac.registered == {}


def test_mac_stop_is_idempotent_and_restartable(mac: MacHotkeyManager, carbon: FakeCarbon) -> None:
    mac.stop()  # never started: nothing to remove
    assert carbon.removed == []
    assert mac.register("t", "ctrl+alt+p", lambda: None)
    mac.stop()
    mac.stop()
    assert carbon.removed == ["handler-ref"]
    assert carbon.registered == {}
    assert mac.register("t", "ctrl+alt+p", lambda: None)
    assert carbon.install_calls == 2


def test_mac_refuses_other_threads(mac: MacHotkeyManager, carbon: FakeCarbon) -> None:
    result: list[bool] = []
    worker = threading.Thread(target=lambda: result.append(mac.register("t", MAC_COMBO, print)))
    worker.start()
    worker.join(2)
    assert result == [False]
    assert carbon.install_calls == 0


def test_mac_keycode_table() -> None:
    codes = hotkeys._MAC_KEYCODES
    assert set(codes) == set(KEYS) - {"f21", "f22", "f23", "f24"}
    assert len(set(codes.values())) == len(codes)
    assert codes["a"] == 0x00
    assert codes["space"] == 0x31
    assert codes["escape"] == 0x35
    assert codes["f1"] == 0x7A
    assert codes["f20"] == 0x5A


def test_fourcc() -> None:
    assert hotkeys._fourcc("keyb") == 0x6B657962
    assert hotkeys._fourcc("hkid") == 0x686B6964
    assert hotkeys._fourcc("----") == 0x2D2D2D2D
    with pytest.raises(ValueError, match="4 bytes"):
        hotkeys._fourcc("abc")


class _Fn:
    """A fake C function: callable and accepts ``argtypes``/``restype`` like ctypes."""

    def __init__(self, impl: Callable[..., Any]) -> None:
        self.impl = impl
        self.argtypes: Any = None
        self.restype: Any = None

    def __call__(self, *args: Any) -> Any:
        return self.impl(*args)


def _value(ref: Any) -> Any:
    """Unwrap a ``c_void_p`` handle (the real library converts it via argtypes)."""
    return getattr(ref, "value", ref)


class FakeCarbonLib:
    """Fake of the Carbon dylib, so the real ctypes binding (structs, the CFUNCTYPE
    trampoline, byref out-parameters) is exercised on every OS."""

    TARGET = 0xABC

    def __init__(self) -> None:
        self.procs: list[Any] = []
        self.hotkeys: dict[int, tuple[int, int, int, int]] = {}
        self.removed: list[int] = []
        self.GetApplicationEventTarget = _Fn(lambda: self.TARGET)
        self.InstallEventHandler = _Fn(self._install)
        self.RemoveEventHandler = _Fn(lambda ref: self.removed.append(_value(ref)) or 0)
        self.RegisterEventHotKey = _Fn(self._register)
        self.UnregisterEventHotKey = _Fn(
            lambda ref: 0 if self.hotkeys.pop(_value(ref), None) else -50
        )
        self.GetEventParameter = _Fn(self._get_parameter)

    def _install(self, target: int, proc: Any, count: int, spec: Any, user: Any, out: Any) -> int:
        assert target == self.TARGET
        assert count == 1
        assert (spec._obj.eventClass, spec._obj.eventKind) == (hotkeys._fourcc("keyb"), 5)
        self.procs.append(proc)
        out._obj.value = 0x1234
        return 0

    def _register(
        self, code: int, mods: int, hk_id: Any, target: int, options: int, out: Any
    ) -> int:
        assert target == self.TARGET
        assert options == 0
        ref = 0x5000 + hk_id.id
        self.hotkeys[ref] = (code, mods, hk_id.signature, hk_id.id)
        out._obj.value = ref
        return 0

    def _get_parameter(
        self, event: int, name: int, kind: int, out_type: Any, size: int, out_size: Any, out: Any
    ) -> int:
        assert (name, kind) == (hotkeys._fourcc("----"), hotkeys._fourcc("hkid"))
        assert size == 8
        if event == 999:
            raise RuntimeError("corrupt event")
        out._obj.signature = hotkeys._fourcc("EyTk")
        out._obj.id = event  # the fake EventRef doubles as the hot key id
        return 0


def test_mac_ctypes_binding_with_fake_library(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import ctypes

    lib = FakeCarbonLib()
    monkeypatch.setattr(ctypes, "CDLL", lambda path: lib)
    manager = MacHotkeyManager()
    fired: list[int] = []
    try:
        assert manager.register("t", "cmd+alt+k", lambda: fired.append(1))
        ((code, mods, signature, hotkey_id),) = lib.hotkeys.values()
        assert (code, mods) == (0x28, 0x0100 | 0x0800)  # kVK_ANSI_K, cmdKey | optionKey
        assert signature == hotkeys._fourcc("EyTk")
        (proc,) = lib.procs
        # Call through the real CFUNCTYPE trampoline, as Carbon would.
        assert proc(None, hotkey_id, None) == 0
        assert fired == [1]
        with caplog.at_level(logging.ERROR, logger="eye_tracker.platform.hotkeys"):
            assert proc(None, 999, None) == -9874  # exceptions never unwind into C
        assert "corrupt event" in caplog.text
        assert manager.unregister("t")
        assert lib.hotkeys == {}
    finally:
        manager.stop()
    assert lib.removed == [0x1234]


def test_mac_carbon_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    def missing(path: str) -> Any:
        raise OSError("no Carbon here")

    monkeypatch.setattr(ctypes, "CDLL", missing)
    manager = MacHotkeyManager()
    assert not manager.register("t", "cmd+alt+k", lambda: None)
    assert manager.note is not None
    manager.stop()


# ================================================================================ X11
KEYSYMS = {
    name: index + 1
    for index, name in enumerate(sorted(set(hotkeys._X_KEYSYM_NAMES.values()) | {"Num_Lock"}))
}
NUMLOCK_KEYCODE = KEYSYMS["Num_Lock"] + 8


class FakeRoot:
    def __init__(self) -> None:
        self.grabs: set[tuple[int, int]] = set()
        self.foreign: set[tuple[int, int]] = set()  # held by "another client"
        self.grab_calls: list[tuple[int, int, bool, int, int]] = []

    def grab_key(
        self,
        keycode: int,
        modifiers: int,
        owner_events: bool,
        pointer_mode: int,
        keyboard_mode: int,
        onerror: Any = None,
    ) -> None:
        self.grab_calls.append((keycode, modifiers, owner_events, pointer_mode, keyboard_mode))
        if (keycode, modifiers) in self.foreign:
            assert onerror is not None
            onerror("BadAccess", None)
            return
        self.grabs.add((keycode, modifiers))

    def ungrab_key(self, keycode: int, modifiers: int, onerror: Any = None) -> None:
        self.grabs.discard((keycode, modifiers))


class FakeDisplay:
    def __init__(self, selectable: bool = False) -> None:
        self.root = FakeRoot()
        self.keymap = {sym: sym + 8 for sym in KEYSYMS.values()}
        # Modifier index 4 (Mod2) holds Num_Lock, as on virtually every keymap.
        self.modmap: list[list[int]] = [[50], [66], [37], [64], [NUMLOCK_KEYCODE], [], [133], []]
        self.events: collections.deque[Any] = collections.deque()
        self.refreshed: list[Any] = []
        self.closed = False
        self.error_handler: Any = None
        self._r = self._w = -1
        if selectable:
            self._r, self._w = os.pipe()
            os.set_blocking(self._r, False)

    # --- python-xlib Display API subset
    def screen(self) -> Any:
        return types.SimpleNamespace(root=self.root)

    def set_error_handler(self, handler: Any) -> None:
        self.error_handler = handler

    def keysym_to_keycode(self, keysym: int) -> int:
        return self.keymap.get(keysym, 0)

    def get_modifier_mapping(self) -> list[list[int]]:
        return self.modmap

    def sync(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def refresh_keyboard_mapping(self, event: Any) -> None:
        self.refreshed.append(event)

    def pending_events(self) -> int:
        if self._r >= 0:
            try:
                while os.read(self._r, 4096):
                    pass
            except BlockingIOError:
                pass
        return len(self.events)

    def next_event(self) -> Any:
        return self.events.popleft()

    def fileno(self) -> int:
        return self._r

    def close(self) -> None:
        self.closed = True
        for fd in (self._r, self._w):
            if fd >= 0:
                os.close(fd)
        self._r = self._w = -1

    # --- test helpers
    def push(self, event: Any) -> None:
        self.events.append(event)
        if self._w >= 0:
            os.write(self._w, b"x")


def key_event(kind: int, keycode: int, state: int, time: int) -> Any:
    return types.SimpleNamespace(type=kind, detail=keycode, state=state, time=time)


def keycode_of(key: str) -> int:
    return KEYSYMS[hotkeys._X_KEYSYM_NAMES[key]] + 8


@pytest.fixture
def fake_keysyms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hotkeys, "_x11_string_to_keysym", lambda name: KEYSYMS.get(name, 0))


@pytest.fixture
def x11(fake_keysyms: None) -> tuple[X11HotkeyManager, FakeDisplay]:
    """An X11 manager wired to a fake display, driven synchronously (no thread)."""
    display = FakeDisplay()
    manager = X11HotkeyManager(display_factory=lambda name: display)
    manager._display = display
    manager._root = display.root
    manager._numlock_mask = manager._find_numlock_mask()
    return manager, display


def bind_x11(manager: X11HotkeyManager, name: str, text: str, callback: Callable[[], None]) -> Any:
    binding = hotkeys._Binding(manager._allocate_id(), name, parse_hotkey(text), callback)
    manager._by_id[binding.id] = binding
    manager._bindings[name] = binding
    return binding


def test_x11_lock_variants() -> None:
    assert hotkeys._x11_lock_variants(16) == (0, 2, 16, 18)
    assert hotkeys._x11_lock_variants(32) == (0, 2, 32, 34)
    assert hotkeys._x11_lock_variants(2) == (0, 2)  # degenerate keymap: no duplicates


def test_x11_modifier_masks() -> None:
    assert hotkeys._x11_modifiers({"ctrl", "alt"}) == 4 | 8
    assert hotkeys._x11_modifiers({"shift", "meta"}) == 1 | 64


def test_x11_keysym_names_cover_every_key() -> None:
    assert set(hotkeys._X_KEYSYM_NAMES) == set(KEYS)


def test_x11_numlock_detection(x11: tuple[X11HotkeyManager, FakeDisplay]) -> None:
    manager, display = x11
    assert manager._numlock_mask == 16
    display.modmap = [[50], [66], [37], [64], [], [NUMLOCK_KEYCODE], [133], []]
    assert manager._find_numlock_mask() == 32
    display.modmap = [[], [], [], [], [], [], [], []]
    assert manager._find_numlock_mask() == 16  # fallback: Mod2


def test_x11_grab_includes_lock_variants(x11: tuple[X11HotkeyManager, FakeDisplay]) -> None:
    manager, display = x11
    binding = bind_x11(manager, "t", "ctrl+alt+p", lambda: None)
    assert manager._grab_binding(binding)
    code = keycode_of("p")
    assert display.root.grabs == {(code, 12), (code, 14), (code, 28), (code, 30)}
    assert all(call[2:] == (True, 1, 1) for call in display.root.grab_calls)
    manager._ungrab_binding(binding.id)
    assert display.root.grabs == set()


def test_x11_bad_access_rolls_back(x11: tuple[X11HotkeyManager, FakeDisplay]) -> None:
    manager, display = x11
    code = keycode_of("p")
    display.root.foreign.add((code, 12 | 16))  # another client owns the NumLock variant
    binding = bind_x11(manager, "t", "ctrl+alt+p", lambda: None)
    assert not manager._grab_binding(binding)
    assert display.root.grabs == set()  # partial grabs released
    assert manager._lookup == {}


def test_x11_missing_key_on_layout(x11: tuple[X11HotkeyManager, FakeDisplay]) -> None:
    manager, display = x11
    display.keymap.pop(KEYSYMS["F24"])
    binding = bind_x11(manager, "t", COMBO, lambda: None)
    assert not manager._grab_binding(binding)


def test_x11_event_dispatch_and_autorepeat(x11: tuple[X11HotkeyManager, FakeDisplay]) -> None:
    manager, _display = x11
    fired: list[int] = []
    binding = bind_x11(manager, "t", "ctrl+alt+p", lambda: fired.append(1))
    assert manager._grab_binding(binding)
    code = keycode_of("p")
    press, release = hotkeys._X_KEY_PRESS, hotkeys._X_KEY_RELEASE

    # Caps Lock + Num Lock active: still matches.
    manager._handle_event(key_event(press, code, 12 | 2 | 16, 1000))
    assert fired == [1]
    # Auto-repeat: release and press share a timestamp → ignored.
    manager._handle_event(key_event(release, code, 12, 1500))
    manager._handle_event(key_event(press, code, 12, 1500))
    manager._handle_event(key_event(release, code, 12, 1530))
    manager._handle_event(key_event(press, code, 12, 1530))
    assert fired == [1]
    # Real release, then a new press later → fires again.
    manager._handle_event(key_event(release, code, 12, 1600))
    manager._handle_event(key_event(press, code, 12, 1800))
    assert fired == [1, 1]
    # Different modifiers or keys are not ours.
    manager._handle_event(key_event(release, code, 12, 1900))
    manager._handle_event(key_event(press, code, 4, 2000))
    manager._handle_event(key_event(press, code + 1, 12, 2100))
    manager._handle_event(types.SimpleNamespace(type=22))  # unrelated event type
    assert fired == [1, 1]


def test_x11_timestamp_wraparound(x11: tuple[X11HotkeyManager, FakeDisplay]) -> None:
    manager, _display = x11
    fired: list[int] = []
    binding = bind_x11(manager, "t", "ctrl+alt+p", lambda: fired.append(1))
    assert manager._grab_binding(binding)
    code = keycode_of("p")
    manager._handle_event(key_event(hotkeys._X_KEY_PRESS, code, 12, 0xFFFFFF00))
    manager._handle_event(key_event(hotkeys._X_KEY_RELEASE, code, 12, 0xFFFFFFFF))
    manager._handle_event(key_event(hotkeys._X_KEY_PRESS, code, 12, 0x00000100))
    assert fired == [1, 1]


def test_x11_mapping_notify_regrabs(x11: tuple[X11HotkeyManager, FakeDisplay]) -> None:
    manager, display = x11
    fired: list[int] = []
    binding = bind_x11(manager, "t", "ctrl+alt+p", lambda: fired.append(1))
    assert manager._grab_binding(binding)
    old = keycode_of("p")
    new = old + 100
    display.keymap[KEYSYMS["p"]] = new  # layout change moved the key
    event = types.SimpleNamespace(type=hotkeys._X_MAPPING_NOTIFY, request=1)
    manager._handle_event(event)
    assert display.refreshed == [event]
    assert {code for code, _ in display.root.grabs} == {new}
    manager._handle_event(key_event(hotkeys._X_KEY_PRESS, new, 12, 10))
    assert fired == [1]
    # Pointer mapping changes are irrelevant.
    manager._handle_event(types.SimpleNamespace(type=hotkeys._X_MAPPING_NOTIFY, request=2))
    assert len(display.refreshed) == 1


def test_x11_error_catcher_protocol() -> None:
    catcher = hotkeys._XErrorCatcher()
    assert catcher("first", None) == 1  # truthy: python-xlib treats the error as handled
    catcher("second", None)
    assert catcher.error == "first"


@pytest.mark.skipif(not IS_POSIX, reason="the X11 loop selects on pipes (POSIX only)")
def test_x11_thread_with_fake_display(fake_keysyms: None) -> None:
    display = FakeDisplay(selectable=True)
    manager = X11HotkeyManager(display_factory=lambda name: display)
    fired = threading.Event()
    threads: list[str] = []

    def callback() -> None:
        threads.append(threading.current_thread().name)
        fired.set()

    try:
        assert manager.register("t", "ctrl+alt+p", callback)
        code = keycode_of("p")
        assert (code, 12) in display.root.grabs
        display.push(key_event(hotkeys._X_KEY_PRESS, code, 12 | 16, 5))
        assert fired.wait(2.0)
        assert threads == ["eye-tracker-hotkeys-x11"]
        assert not manager.register("dup", "ctrl+alt+p", lambda: None)
        assert manager.unregister("t")
        assert display.root.grabs == set()
        assert manager.register("t", "ctrl+alt+p", callback)
    finally:
        manager.stop()
        manager.stop()
    assert display.closed
    assert display.root.grabs == set()
    assert manager._thread is None


@pytest.mark.skipif(not IS_POSIX, reason="the X11 loop selects on pipes (POSIX only)")
def test_x11_unreachable_display() -> None:
    def fail(name: str | None) -> Any:
        raise ConnectionError("no X server")

    manager = X11HotkeyManager(display_factory=fail)
    try:
        assert not manager.register("t", "ctrl+alt+p", lambda: None)
        assert manager.note is not None
        assert "X server" in manager.note
    finally:
        manager.stop()


def _has_xlib() -> bool:
    try:
        import Xlib  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.environ.get("DISPLAY") or not _has_xlib(),
    reason="needs a real X11 display and python-xlib",
)
def test_x11_live_register() -> None:
    manager = X11HotkeyManager()
    try:
        assert manager.register("t", COMBO, lambda: None)
        assert manager.registered == {"t": parse_hotkey(COMBO)}
    finally:
        manager.stop()
        manager.stop()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_mac_live_register_smoke() -> None:
    # Isolated in a subprocess so a Carbon problem on a headless runner cannot take
    # the whole test session down.
    code = (
        "from eye_tracker.platform.hotkeys import MacHotkeyManager, parse_hotkey\n"
        "m = MacHotkeyManager()\n"
        "ok = m.register('t', parse_hotkey('ctrl+alt+shift+f20'), lambda: None)\n"
        "m.stop(); m.stop()\n"
        "print(ok)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in {"True", "False"}


# ====================================================================== Windows live
@pytest.fixture
def win() -> Iterator[WindowsHotkeyManager]:
    manager = WindowsHotkeyManager()
    try:
        yield manager
    finally:
        manager.stop()


def post_hotkey(manager: WindowsHotkeyManager, name: str) -> None:
    """Deliver WM_HOTKEY to the manager's own thread (no input is synthesised)."""
    api = hotkeys._win32()
    binding = manager._bindings[name]
    assert api.PostThreadMessageW(manager._thread_id, hotkeys._WM_HOTKEY, binding.id, 0)


windows_only = pytest.mark.skipif(not IS_WINDOWS, reason="Windows only")


@windows_only
def test_windows_factory_returns_native_manager() -> None:
    manager = create_hotkey_manager()
    assert isinstance(manager, WindowsHotkeyManager)
    manager.stop()


@windows_only
def test_windows_live_register_dispatch_and_stop(win: WindowsHotkeyManager) -> None:
    fired = threading.Event()
    threads: list[str] = []

    def callback() -> None:
        threads.append(threading.current_thread().name)
        fired.set()

    assert win.register("t", parse_hotkey(COMBO), callback)
    assert win.registered == {"t": parse_hotkey(COMBO)}
    thread = win._thread
    assert thread is not None
    assert thread.is_alive()
    post_hotkey(win, "t")
    assert fired.wait(2.0)
    assert threads == ["eye-tracker-hotkeys-win32"]
    win.stop()
    assert not thread.is_alive()
    assert win.registered == {}
    win.stop()  # idempotent


@windows_only
def test_windows_double_registration(win: WindowsHotkeyManager) -> None:
    fired: list[str] = []
    done = threading.Event()

    def second() -> None:
        fired.append("second")
        done.set()

    assert win.register("a", COMBO, lambda: fired.append("first"))
    assert win.register("a", COMBO, second)  # same name + combo: callback replaced
    assert not win.register("b", COMBO, lambda: None)  # combo owned by "a"
    assert set(win.registered) == {"a"}
    post_hotkey(win, "a")
    assert done.wait(2.0)
    assert fired == ["second"]


@windows_only
def test_windows_combo_taken_by_another_owner() -> None:
    first, second = WindowsHotkeyManager(), WindowsHotkeyManager()
    try:
        assert first.register("t", COMBO, lambda: None)
        # The OS refuses a combination that another thread/app already registered.
        assert not second.register("t", COMBO, lambda: None)
        assert second.registered == {}
        first.stop()
        assert second.register("t", COMBO, lambda: None)  # released by stop()
    finally:
        first.stop()
        second.stop()


@windows_only
def test_windows_unregister_releases_os_registration(win: WindowsHotkeyManager) -> None:
    other = WindowsHotkeyManager()
    try:
        assert win.register("t", COMBO, lambda: None)
        assert win.unregister("t")
        assert other.register("t", COMBO, lambda: None)
        other.unregister_all()
        assert win.register("t", COMBO, lambda: None)
    finally:
        other.stop()


@windows_only
def test_windows_callback_may_register_and_stop(win: WindowsHotkeyManager) -> None:
    results: list[bool] = []
    done = threading.Event()

    def register_more() -> None:
        results.append(win.register("second", COMBO_2, lambda: None))
        done.set()

    assert win.register("first", COMBO, register_more)
    post_hotkey(win, "first")
    assert done.wait(2.0)
    assert results == [True]
    assert set(win.registered) == {"first", "second"}

    stopped = threading.Event()

    def stop_from_callback() -> None:
        win.stop()
        stopped.set()

    assert win.register("first", COMBO, stop_from_callback)
    thread = win._thread
    assert thread is not None
    post_hotkey(win, "first")
    assert stopped.wait(2.0)
    thread.join(2.0)
    assert not thread.is_alive()
    assert win.registered == {}
    assert win.register("again", COMBO, lambda: None)  # restarts cleanly


@windows_only
def test_windows_callback_exception_keeps_thread_alive(
    win: WindowsHotkeyManager, caplog: pytest.LogCaptureFixture
) -> None:
    def boom() -> None:
        raise RuntimeError("callback failure")

    done = threading.Event()
    assert win.register("bad", COMBO, boom)
    assert win.register("good", COMBO_2, done.set)
    with caplog.at_level(logging.ERROR, logger="eye_tracker.platform.hotkeys"):
        post_hotkey(win, "bad")
        post_hotkey(win, "good")
        assert done.wait(2.0)
    assert "callback failure" in caplog.text
    thread = win._thread
    assert thread is not None
    assert thread.is_alive()


@windows_only
def test_windows_start_prewarms_thread(win: WindowsHotkeyManager) -> None:
    win.start()
    thread = win._thread
    assert thread is not None
    assert thread.is_alive()
    win.stop()
    assert not thread.is_alive()
