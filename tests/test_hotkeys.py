"""Tests for global hotkey parsing, formatting and the native managers.

Nothing here synthesises key presses. On Windows the live tests register an
unusual combination and deliver ``WM_HOTKEY`` to the manager's own thread with
``PostThreadMessageW``; macOS and X11 code paths run against fakes (plus an
isolated live smoke test on those systems).
"""

from __future__ import annotations

import array
import collections
import logging
import os
import subprocess
import sys
import threading
import types
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

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
        ("shift+escape", "shift+escape"),
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
        # Shift+navigation extends selections (and scrolls terminals) everywhere.
        "shift+left",
        "shift+right",
        "shift+up",
        "shift+down",
        "shift+home",
        "shift+end",
        "shift+pageup",
        "shift+pagedown",
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


@pytest.mark.parametrize(
    ("platform", "modifiers"),
    [
        # Windows: Ctrl+Alt is AltGr, so the Win key keeps the defaults from typing.
        ("win32", {"ctrl", "alt", "meta"}),
        # Linux: Ctrl+Alt+T opens a terminal, and Ctrl+Alt+Shift cannot be pressed with
        # an Alt+Shift / Ctrl+Shift layout switch and is a JetBrains shortcut.
        ("linux", {"ctrl", "alt", "meta"}),
        ("freebsd14", {"ctrl", "alt", "meta"}),
        # macOS: ⌃⌥T and ⌃⌥C are Rectangle's and Magnet's defaults.
        ("darwin", {"ctrl", "alt", "meta"}),
    ],
)
def test_default_settings_hotkeys_per_platform(
    monkeypatch: pytest.MonkeyPatch, platform: str, modifiers: set[str]
) -> None:
    from eye_tracker.config import HotkeySettings, Settings

    monkeypatch.setattr(sys, "platform", platform)
    defaults = HotkeySettings()
    parsed = [
        parse_hotkey(text)
        for text in (defaults.toggle_tracking, defaults.toggle_privacy, defaults.recalibrate)
    ]
    assert [p.key for p in parsed] == ["t", "p", "c"]
    assert all(p.modifiers == frozenset(modifiers) for p in parsed)
    assert Settings().hotkeys == defaults
    # Loading a file that lacks the hotkeys keeps the platform defaults.
    assert Settings.from_dict({"hotkeys": {"enabled": True}}).hotkeys == defaults


def test_windows_defaults_never_collide_with_altgr(monkeypatch: pytest.MonkeyPatch) -> None:
    from eye_tracker.config import HotkeySettings

    monkeypatch.setattr(sys, "platform", "win32")
    defaults = HotkeySettings()
    probe = FakeLayoutProbe(everything="x")  # a layout on which every AltGr key types
    for text in (defaults.toggle_tracking, defaults.toggle_privacy, defaults.recalibrate):
        hotkey = parse_hotkey(text)
        vk = hotkeys._win_virtual_key(hotkey.key)
        assert vk is not None
        assert hotkeys._win_altgr_conflict(hotkey, vk, probe) is None, text
    assert probe.queries == []  # never even asked: Win+AltGr is not a typing chord


def test_linux_defaults_avoid_desktop_shortcuts(monkeypatch: pytest.MonkeyPatch) -> None:
    from eye_tracker.config import HotkeySettings

    monkeypatch.setattr(sys, "platform", "linux")
    defaults = HotkeySettings()
    taken = {parse_hotkey("ctrl+alt+t")} | {parse_hotkey(f"ctrl+alt+f{n}") for n in range(1, 13)}
    for text in (defaults.toggle_tracking, defaults.toggle_privacy, defaults.recalibrate):
        assert parse_hotkey(text) not in taken


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


def test_wayland_notes_name_the_commands_of_this_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (r2-docs-01): the release packages put no "eye-tracker" command on
    the PATH, and a desktop shortcut bound to a missing command fails silently."""
    from eye_tracker import cli

    appimage = "/home/me/Apps/Eye Tracker.AppImage"
    monkeypatch.setattr(
        cli, "cli_command_text", lambda *args: cli.format_command([appimage, *args])
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    wayland = create_hotkey_manager().note

    monkeypatch.setenv("DISPLAY", ":0")  # XWayland
    monkeypatch.setitem(sys.modules, "Xlib", types.ModuleType("Xlib"))
    xwayland = create_hotkey_manager().note

    for note in (wayland, xwayland):
        assert note is not None
        assert "eye-tracker ctl" not in note
        for action in ("toggle", "privacy-toggle", "calibrate"):
            assert f"'{cli.format_command([appimage, 'ctl', action])}'" in note
    assert wayland is not None
    assert wayland.startswith("Wayland does not allow applications to register global hotkeys.")
    assert xwayland is not None
    assert "may only fire while an X11 window has focus" in xwayland


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
    # … a character the layout only types with Shift/AltGr has no key of its own:
    # the US virtual key would be another key (Turkish Q: 0xDB is 'ğ', not '[').
    assert hotkeys._win_virtual_key("period", lambda ch: 0x0137) is None
    assert hotkeys._win_virtual_key("bracketleft", lambda ch: 0x0638) is None  # AltGr+8
    # A character missing from the layout (Cyrillic) falls back to the US position …
    assert hotkeys._win_virtual_key("period", lambda ch: -1) == 0xBE
    assert hotkeys._win_virtual_key("period", lambda ch: -1, lambda vk: 0x34) == 0xBE
    # … unless no physical key produces that virtual key on this layout.
    assert hotkeys._win_virtual_key("equal", lambda ch: -1, lambda vk: 0) is None
    # A failing layout query is treated like "not on the layout".
    assert hotkeys._win_virtual_key("period", _raise_os_error, lambda vk: 0x34) == 0xBE
    # Letters never consult the layout.
    assert hotkeys._win_virtual_key("q", lambda ch: 0x99, lambda vk: 0) == ord("Q")


def _raise_os_error(*_args: Any) -> int:
    raise OSError("layout query failed")


def test_windows_modifier_mask() -> None:
    assert hotkeys._win_modifiers({"ctrl", "alt"}) == 0x0003
    assert hotkeys._win_modifiers({"shift", "meta"}) == 0x000C


# ================================================== Windows AltGr (Ctrl+Alt) collisions
TURKISH_Q, POLISH_PROGRAMMERS, US = 0x041F041F, 0x04150415, 0x04090409
VK_T, VK_P, VK_C, VK_A, VK_8, VK_OEM_1 = 0x54, 0x50, 0x43, 0x41, 0x38, 0xBA

# What AltGr(+Shift) types, as read from the layout DLL tables (kbdtuq.dll,
# kbdpl1.dll) and confirmed with ToUnicodeEx on an installed Turkish Q layout.
LAYOUT_TABLES: dict[int, dict[tuple[int, bool], tuple[str, bool]]] = {
    TURKISH_Q: {
        (VK_T, False): ("₺", False),
        (VK_8, False): ("[", False),
        (VK_A, True): ("Æ", False),
        (VK_OEM_1, False): ("´", True),  # dead key
    },
    POLISH_PROGRAMMERS: {(VK_C, False): ("ć", False), (VK_C, True): ("Ć", False)},
    US: {},
}


class FakeLayoutProbe:
    """Installed keyboard layouts as data (the real probe asks user32)."""

    NAMES: ClassVar[dict[int, str]] = {
        TURKISH_Q: "Turkish Q",
        POLISH_PROGRAMMERS: "Polish (Programmers)",
        US: "US",
    }

    def __init__(
        self,
        tables: dict[int, dict[tuple[int, bool], tuple[str, bool]]] | None = None,
        everything: str | None = None,
    ) -> None:
        self.tables = tables if tables is not None else {US: {}}
        self.everything = everything
        self.queries: list[tuple[int, bool, int]] = []

    def layouts(self) -> list[int]:
        return list(self.tables)

    def altgr_text(self, vk: int, shift: bool, hkl: int) -> tuple[str, bool] | None:
        self.queries.append((vk, shift, hkl))
        if self.everything is not None:
            return self.everything, False
        return self.tables[hkl].get((vk, shift))

    def layout_name(self, hkl: int) -> str:
        return self.NAMES.get(hkl, hex(hkl))


def altgr_conflict(text: str, probe: FakeLayoutProbe) -> str | None:
    hotkey = parse_hotkey(text)
    vk = hotkeys._win_virtual_key(hotkey.key)
    assert vk is not None
    return hotkeys._win_altgr_conflict(hotkey, vk, probe)


def test_altgr_conflict_catches_the_old_defaults() -> None:
    probe = FakeLayoutProbe(LAYOUT_TABLES)
    assert altgr_conflict("ctrl+alt+t", probe) == (
        "Ctrl+Alt+T is AltGr+T and types '₺' on the Turkish Q keyboard layout"
    )
    assert altgr_conflict("ctrl+alt+c", probe) == (
        "Ctrl+Alt+C is AltGr+C and types 'ć' on the Polish (Programmers) keyboard layout"
    )
    assert altgr_conflict("ctrl+alt+p", probe) is None  # nothing on these three layouts
    # Every installed layout is asked, because RegisterHotKey fires on any of them.
    assert {hkl for _vk, _shift, hkl in probe.queries} == set(LAYOUT_TABLES)


def test_altgr_conflict_with_shift_and_dead_keys() -> None:
    probe = FakeLayoutProbe(LAYOUT_TABLES)
    message = altgr_conflict("ctrl+alt+shift+c", probe)
    assert message is not None
    assert "AltGr+Shift+C" in message
    assert "'Ć'" in message
    assert all(shift for _vk, shift, _hkl in probe.queries)
    message = altgr_conflict("ctrl+alt+8", probe)
    assert message is not None
    assert "'['" in message
    dead = hotkeys._win_altgr_conflict(parse_hotkey("ctrl+alt+semicolon"), VK_OEM_1, probe)
    assert dead is not None
    assert "the dead key '´'" in dead


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ć", "'ć'"),
        ("ab", "'ab'"),
        ("'", '"\'"'),
        ("e\u0301", "'é'"),  # composed first
        # Invisible characters are named, never shown as Python escapes like '\xa0'.
        ("\xa0", "no-break space (U+00A0)"),
        ("\u202f", "narrow no-break space (U+202F)"),
        (" ", "space (U+0020)"),
        ("\u200b", "zero width space (U+200B)"),
        ("\u0301", "combining acute accent (U+0301)"),  # would sit on the quote sign
        ("\ue000", "U+E000"),  # private use: no name
        ("x\xa0", "latin small letter x (U+0078) + no-break space (U+00A0)"),
        ("", "nothing"),
    ],
)
def test_typed_characters_are_described_readably(text: str, expected: str) -> None:
    assert hotkeys._describe_typed(text) == expected


def test_altgr_conflict_messages_read_well() -> None:
    assert altgr_conflict("ctrl+alt+space", FakeLayoutProbe(everything="\xa0")) == (
        "Ctrl+Alt+Space is AltGr+Space and types no-break space (U+00A0) on the US keyboard layout"
    )
    # A punctuation key does not run into the sentence ("is AltGr+,, which types").
    assert altgr_conflict("ctrl+alt+comma", FakeLayoutProbe(everything="ç")) == (
        "Ctrl+Alt+, is AltGr+, and types 'ç' on the US keyboard layout"
    )


@pytest.mark.parametrize(
    "text", ["alt+t", "ctrl+t", "ctrl+shift+t", "ctrl+alt+meta+t", "alt+shift+meta+c"]
)
def test_altgr_conflict_only_checks_ctrl_alt_without_win(text: str) -> None:
    probe = FakeLayoutProbe(everything="x")
    assert altgr_conflict(text, probe) is None
    assert probe.queries == []


def test_altgr_conflict_function_keys_never_type() -> None:
    probe = FakeLayoutProbe(LAYOUT_TABLES)
    assert altgr_conflict("ctrl+alt+f9", probe) is None
    assert altgr_conflict("ctrl+alt+shift+f24", probe) is None


def test_windows_manager_layout_conflict_uses_probe() -> None:
    manager = WindowsHotkeyManager(layout_probe=FakeLayoutProbe(LAYOUT_TABLES))
    conflict = manager.layout_conflict("ctrl+alt+t")
    assert conflict is not None
    assert "Turkish Q" in conflict
    assert manager.layout_conflict(parse_hotkey("ctrl+alt+meta+t")) is None
    with pytest.raises(ValueError, match="unsupported key"):
        manager.layout_conflict("ctrl+alt+nosuchkey")

    class BrokenProbe(FakeLayoutProbe):
        def layouts(self) -> list[int]:
            raise OSError("user32 is gone")

    # Advisory only: a failing query never breaks the caller.
    assert WindowsHotkeyManager(layout_probe=BrokenProbe()).layout_conflict("ctrl+alt+t") is None


def test_base_manager_layout_conflict() -> None:
    manager = HotkeyManager()
    assert manager.layout_conflict("ctrl+alt+t") is None
    with pytest.raises(ValueError, match="needs a key"):
        manager.layout_conflict("ctrl+alt")


class FakeUser32:
    """Just enough of :class:`hotkeys._Win32Api` for :class:`hotkeys._Win32LayoutProbe`."""

    def __init__(self, results: dict[tuple[int, int, bool], tuple[int, str]]) -> None:
        import ctypes

        self.HKL = ctypes.c_void_p
        self.c_ubyte = ctypes.c_ubyte
        self.create_unicode_buffer = ctypes.create_unicode_buffer
        self.results = results  # (hkl, vk, shift) → (ToUnicodeEx result, buffer text)
        self.flags: list[int] = []
        self.hkls = [TURKISH_Q, US]

    def GetKeyboardLayoutList(self, count: int, handles: Any) -> int:
        if not count:
            return len(self.hkls)
        for index, hkl in enumerate(self.hkls[:count]):
            handles[index] = hkl
        return min(count, len(self.hkls))

    def MapVirtualKeyExW(self, vk: int, kind: int, hkl: int) -> int:
        assert kind == 0  # MAPVK_VK_TO_VSC
        return 0 if vk == 0xFF else 0x14

    def ToUnicodeEx(
        self, vk: int, scan: int, state: Any, buf: Any, size: int, flags: int, hkl: int
    ) -> int:
        self.flags.append(flags)
        assert scan == 0x14
        assert size == len(buf)
        for held in (0x11, 0xA2, 0x12, 0xA5):  # VK_CONTROL, VK_LCONTROL, VK_MENU, VK_RMENU
            assert state[held] == 0x80
        shift = state[0x10] == 0x80
        assert (state[0xA0] == 0x80) is shift
        count, text = self.results.get((hkl, vk, shift), (0, ""))
        for index, char in enumerate(text):
            buf[index] = char
        return count

    def GetLocaleInfoW(self, lcid: int, kind: int, buf: Any, size: int) -> int:
        assert kind == 0x72  # LOCALE_SENGLISHDISPLAYNAME
        if lcid != 0x041F:
            return 0
        buf.value = "Turkish (Türkiye)"
        return len(buf.value) + 1


def test_win32_layout_probe_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeUser32(
        {
            (TURKISH_Q, VK_T, False): (1, "₺"),
            (TURKISH_Q, VK_OEM_1, False): (-1, "´"),
            (TURKISH_Q, 0x0D, False): (1, "\r"),  # control characters are not typing
            (TURKISH_Q, VK_A, True): (1, "Æ"),
        }
    )
    probe = hotkeys._Win32LayoutProbe(api)  # type: ignore[arg-type]
    assert probe.layouts() == [TURKISH_Q, US]
    assert probe.altgr_text(VK_T, False, TURKISH_Q) == ("₺", False)
    assert probe.altgr_text(VK_T, False, US) is None
    assert probe.altgr_text(VK_OEM_1, False, TURKISH_Q) == ("´", True)
    assert probe.altgr_text(0x0D, False, TURKISH_Q) is None
    assert probe.altgr_text(VK_A, True, TURKISH_Q) == ("Æ", False)
    assert probe.altgr_text(0xFF, False, TURKISH_Q) is None  # no key makes this VK
    # Flag 0x4: the query never changes the keyboard state (no dead key is armed).
    assert set(api.flags) == {0x4}
    monkeypatch.setattr(hotkeys, "_win_layout_text", lambda hkl: None)
    assert probe.layout_name(TURKISH_Q) == "Turkish (Türkiye)"
    assert probe.layout_name(0x12345678) == "0x12345678"
    monkeypatch.setattr(hotkeys, "_win_layout_text", lambda hkl: "Turkish Q")
    assert probe.layout_name(TURKISH_Q) == "Turkish Q"


# ======================================================== generic threaded machinery
class LoopbackManager(hotkeys._ThreadedHotkeyManager):
    """A portable threaded manager whose "OS" is a dict, to test the shared plumbing."""

    name = "loopback"

    def __init__(self) -> None:
        super().__init__()
        self._wake_event = threading.Event()
        self.os_registered: dict[int, Hotkey] = {}
        self.taken: set[Hotkey] = set()
        self.reasons: dict[Hotkey, str] = {}  # taken combos that explain themselves
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
        if binding.hotkey in self.reasons:
            return self._reject(binding, self.reasons[binding.hotkey])
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


def test_last_error_explains_failures(
    loopback: LoopbackManager, caplog: pytest.LogCaptureFixture
) -> None:
    assert loopback.last_error("t") is None  # never registered
    assert not loopback.register("t", "ctrl+alt+nosuchkey", lambda: None)
    error = loopback.last_error("t")
    assert error is not None
    assert "unsupported key" in error
    # A native hook that explains itself (e.g. an AltGr collision) wins …
    reason = "Ctrl+Alt+T is AltGr+T, which types '₺' on the Turkish Q keyboard layout"
    loopback.reasons[parse_hotkey("ctrl+alt+t")] = reason
    with caplog.at_level(logging.WARNING, logger="eye_tracker.platform.hotkeys"):
        assert not loopback.register("t", "ctrl+alt+t", lambda: None)
    assert loopback.last_error("t") == reason
    assert reason in caplog.text
    # … otherwise a generic message is recorded.
    loopback.taken.add(parse_hotkey(COMBO))
    assert not loopback.register("t", COMBO, lambda: None)
    assert loopback.last_error("t") == "Ctrl+Alt+Shift+F24 could not be registered"
    # A clash inside this manager names the other action.
    assert loopback.register("a", COMBO_2, lambda: None)
    assert not loopback.register("b", COMBO_2, lambda: None)
    assert loopback.last_error("b") == "Ctrl+Alt+Shift+F21 is already used for a"
    # Success clears the error.
    assert loopback.register("t", "ctrl+alt+shift+f20", lambda: None)
    assert loopback.last_error("t") is None
    assert loopback.last_error("a") is None


def test_unsupported_manager_last_error() -> None:
    manager = HotkeyManager(note="Wayland does not allow global hotkeys.")
    assert not manager.register("t", COMBO, lambda: None)
    assert manager.last_error("t") == "Wayland does not allow global hotkeys."
    assert not HotkeyManager().register("t", COMBO, lambda: None)


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
def _mac_layout(**overrides: str) -> dict[int, str]:
    """A Mac layout's base layer ``{keycode: char}``: US plus ``overrides`` by US key name."""
    layout = {code: char for char, code in hotkeys._MAC_ANSI_CHARS.items()}
    for key, char in overrides.items():
        layout[hotkeys._MAC_KEYCODES[key]] = char
    return layout


def _without(layout: dict[int, str], *chars: str) -> dict[int, str]:
    return {code: char for code, char in layout.items() if char not in chars}


MAC_US = _mac_layout()
# French AZERTY: A/Q, Z/W and M swapped around; the number row types &é"'(§è!çà.
MAC_AZERTY = _mac_layout(
    a="q", q="a", w="z", z="w", semicolon="m", m=",", comma=";", period=":", slash="=",
    **{"1": "&", "2": "é", "3": '"', "4": "'", "5": "(", "6": "§", "7": "è", "8": "!",
       "9": "ç", "0": "à"},
    minus=")", equal="-", bracketleft="^", bracketright="$", quote="ù", backslash="`", grave="<",
)  # fmt: skip
# Dvorak: the default ⌃⌥T/P/C sit on the keys at the US K, R and I positions.
MAC_DVORAK = _mac_layout(
    q="'", w=",", e=".", r="p", t="y", y="f", u="g", i="c", o="r", p="l", bracketleft="/",
    bracketright="=", s="o", d="e", f="u", g="i", h="d", j="h", k="t", l="n", semicolon="s",
    quote="-", z=";", x="q", c="j", v="k", b="x", n="b", comma="w", period="v", slash="z",
    equal="]", minus="[",
)  # fmt: skip
# Russian: Cyrillic letters, so Latin shortcuts come from the ASCII-capable layout.
MAC_RUSSIAN = _mac_layout(
    a="ф", s="ы", d="в", f="а", t="е", p="з", c="с", k="л", comma="б", period="ю", slash="."
)


class FakeCarbon:
    """Stand-in for the ctypes Carbon binding."""

    def __init__(
        self,
        layout: dict[int, str] | None = None,
        ascii_layout: dict[int, str] | None = None,
        modified: dict[int, dict[int, str]] | None = None,
    ) -> None:
        self.handler: Callable[[Any], int] | None = None
        self.install_calls = 0
        self.removed: list[Any] = []
        self.registered: dict[str, tuple[int, int, int, int]] = {}
        self.reject: dict[tuple[int, int], int] = {}
        self._next = 0
        # Keyboard layout: base layer of the current / ASCII-capable layout and the
        # layers typed with Option(+Shift), keyed by Carbon modifier mask.
        self.layout = layout
        self.ascii_layout = ascii_layout if ascii_layout is not None else layout
        self.modified = modified or {}
        self.layout_callback: Callable[[], None] | None = None
        self.stopped_observing = 0

    def layout_characters(
        self, modifiers: int = 0, *, ascii_capable: bool = False
    ) -> dict[int, str] | None:
        if modifiers:
            return self.modified.get(modifiers, {})
        return self.ascii_layout if ascii_capable else self.layout

    def observe_layout_changes(self, callback: Callable[[], None]) -> bool:
        self.layout_callback = callback
        return True

    def stop_observing_layout_changes(self) -> None:
        self.layout_callback = None
        self.stopped_observing += 1

    def switch_layout(
        self, layout: dict[int, str], ascii_layout: dict[int, str] | None = None
    ) -> None:
        self.layout = layout
        self.ascii_layout = ascii_layout if ascii_layout is not None else layout
        assert self.layout_callback is not None
        self.layout_callback()

    def keycode_for(self, hotkey_id: int) -> int | None:
        """Key code at which Carbon currently holds the hot key ``hotkey_id``."""
        codes = [
            code for code, _mods, _sig, hk_id in self.registered.values() if hk_id == hotkey_id
        ]
        assert len(codes) <= 1
        return codes[0] if codes else None

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
    # Registered exclusively, so this means another application owns it.
    assert mac.last_error("t") == "⌃⌥P is already in use by another application"
    assert not mac.register("t", "ctrl+alt+f24", lambda: None)  # no F24 on a Mac
    assert mac.last_error("t") == "⌃⌥F24: Mac keyboards have no F24"
    carbon.reject[(0x23, 0x1800)] = -50
    assert not mac.register("t", "ctrl+alt+p", lambda: None)
    assert mac.last_error("t") == "RegisterEventHotKey(⌃⌥P) failed (OSStatus -50)"
    assert mac.registered == {}


def mac_keycode(mac: MacHotkeyManager, carbon: FakeCarbon, name: str) -> int | None:
    return carbon.keycode_for(mac._bindings[name].id)


def test_mac_letters_follow_the_keyboard_layout() -> None:
    carbon = FakeCarbon(MAC_AZERTY)
    mac = MacHotkeyManager(carbon=carbon)
    try:
        for name, text in [("a", "ctrl+alt+a"), ("m", "ctrl+alt+m"), ("comma", "ctrl+alt+,")]:
            assert mac.register(name, text, lambda: None)
        assert mac.register("one", "ctrl+alt+1", lambda: None)
        assert mac.register("t", "ctrl+alt+t", lambda: None)
        codes = {name: mac_keycode(mac, carbon, name) for name in mac.registered}
        # The key labelled A on AZERTY is at the US Q position, and so on.
        assert codes["a"] == hotkeys._MAC_KEYCODES["q"]
        assert codes["m"] == hotkeys._MAC_KEYCODES["semicolon"]
        assert codes["comma"] == hotkeys._MAC_KEYCODES["m"]
        # Digits need Shift on AZERTY: the number row keeps its (labelled) US position.
        assert codes["one"] == hotkeys._MAC_KEYCODES["1"]
        assert codes["t"] == hotkeys._MAC_KEYCODES["t"]
        # '/' needs modifiers on AZERTY: no key is "the / key", so nothing is guessed.
        assert not mac.register("slash", "ctrl+alt+/", lambda: None)
        assert mac.last_error("slash") == (
            "⌃⌥/: no key types '/' without modifiers on the current keyboard layout"
        )
    finally:
        mac.stop()
    assert carbon.stopped_observing == 1


def test_mac_default_hotkeys_on_dvorak_use_the_labelled_keys() -> None:
    carbon = FakeCarbon(MAC_DVORAK)
    mac = MacHotkeyManager(carbon=carbon)
    try:
        for name, key in [("toggle_tracking", "t"), ("toggle_privacy", "p"), ("recalibrate", "c")]:
            assert mac.register(name, f"ctrl+alt+{key}", lambda: None)
        assert mac_keycode(mac, carbon, "toggle_tracking") == hotkeys._MAC_KEYCODES["k"]
        assert mac_keycode(mac, carbon, "toggle_privacy") == hotkeys._MAC_KEYCODES["r"]
        assert mac_keycode(mac, carbon, "recalibrate") == hotkeys._MAC_KEYCODES["i"]
    finally:
        mac.stop()


def test_mac_non_latin_layout_uses_the_ascii_capable_layout() -> None:
    carbon = FakeCarbon(MAC_RUSSIAN, ascii_layout=MAC_DVORAK)
    mac = MacHotkeyManager(carbon=carbon)
    try:
        assert mac.register("t", "ctrl+alt+t", lambda: None)
        assert mac_keycode(mac, carbon, "t") == hotkeys._MAC_KEYCODES["k"]  # Dvorak 't'
        assert mac.register("period", "ctrl+alt+.", lambda: None)
        # The current layout wins where it has the character (Russian '.' at US '/').
        assert mac_keycode(mac, carbon, "period") == hotkeys._MAC_KEYCODES["slash"]
    finally:
        mac.stop()


def test_mac_without_layout_data_uses_us_positions() -> None:
    carbon = FakeCarbon(layout=None)
    mac = MacHotkeyManager(carbon=carbon)
    try:
        assert mac.register("a", "ctrl+alt+a", lambda: None)
        assert mac.register("slash", "ctrl+alt+/", lambda: None)
        assert mac_keycode(mac, carbon, "a") == 0x00
        assert mac_keycode(mac, carbon, "slash") == 0x2C
    finally:
        mac.stop()


def test_mac_registrations_follow_layout_switches() -> None:
    carbon = FakeCarbon(MAC_US)
    mac = MacHotkeyManager(carbon=carbon)
    fired: list[str] = []
    try:
        assert mac.register("a", "ctrl+alt+a", lambda: fired.append("a"))
        assert mac.register("t", "ctrl+alt+t", lambda: None)
        assert mac.register("slash", "ctrl+alt+/", lambda: None)
        t_ref = next(ref for ref, v in carbon.registered.items() if v[3] == mac._bindings["t"].id)
        carbon.switch_layout(MAC_AZERTY)
        assert mac_keycode(mac, carbon, "a") == hotkeys._MAC_KEYCODES["q"]
        assert t_ref in carbon.registered  # unchanged key: registration left alone
        # '/' has no key of its own on AZERTY: the hotkey is inactive, not forgotten.
        assert set(mac.registered) == {"a", "t"}
        assert mac_keycode(mac, carbon, "slash") is None
        error = mac.last_error("slash")
        assert error is not None
        assert "no key types '/'" in error
        assert carbon.press(hotkeys._MAC_KEYCODES["q"], 0x1800) == 0
        assert fired == ["a"]
        carbon.switch_layout(MAC_US)  # back again: everything is restored
        assert set(mac.registered) == {"a", "t", "slash"}
        assert mac_keycode(mac, carbon, "slash") == 0x2C
        assert mac_keycode(mac, carbon, "a") == 0x00
        assert mac.last_error("slash") is None
        # Re-registering the inactive-then-restored name is a no-op like any other.
        assert mac.register("slash", "ctrl+alt+/", lambda: None)
        assert len(carbon.registered) == 3
    finally:
        mac.stop()
    assert carbon.registered == {}


def test_mac_register_retries_an_inactive_binding() -> None:
    carbon = FakeCarbon(MAC_US)
    mac = MacHotkeyManager(carbon=carbon)
    try:
        assert mac.register("slash", "ctrl+alt+/", lambda: None)
        carbon.switch_layout(MAC_AZERTY)
        assert mac.registered == {}
        carbon.layout = carbon.ascii_layout = MAC_US  # changed without a notification
        mac._keymap = mac._read_keymap()
        assert mac.register("slash", "ctrl+alt+/", lambda: None)  # not the no-op path
        assert mac.registered == {"slash": parse_hotkey("ctrl+alt+/")}
    finally:
        mac.stop()


def test_mac_option_combinations_that_type_are_refused() -> None:
    option, shift = 0x0800, 0x0200
    e_code = hotkeys._MAC_KEYCODES["e"]
    carbon = FakeCarbon(MAC_US, modified={option: {e_code: "´"}, option | shift: {e_code: "´"}})
    mac = MacHotkeyManager(carbon=carbon)
    try:
        assert not mac.register("e", "alt+e", lambda: None)
        assert mac.last_error("e") == "⌥E types '´' on the current keyboard layout"
        assert not mac.register("e", "alt+shift+e", lambda: None)
        assert mac.layout_conflict("alt+e") == "⌥E types '´' on the current keyboard layout"
        # Control or Command switch the typing layer off.
        assert mac.layout_conflict("ctrl+alt+e") is None
        assert mac.register("e", "ctrl+alt+e", lambda: None)
        assert mac.register("f5", "alt+f5", lambda: None)  # types nothing
        assert mac.layout_conflict("alt+r") is None  # nothing on this fake's Option+R
    finally:
        mac.stop()


def test_mac_option_space_names_the_no_break_space() -> None:
    """⌥Space (Alfred's default) types U+00A0 on US/ABC: say so without escapes."""
    space = hotkeys._MAC_KEYCODES["space"]
    carbon = FakeCarbon(MAC_US, modified={0x0800: {space: "\xa0"}})
    mac = MacHotkeyManager(carbon=carbon)
    try:
        assert not mac.register("s", "alt+space", lambda: None)
        assert mac.last_error("s") == (
            "⌥Space types no-break space (U+00A0) on the current keyboard layout"
        )
    finally:
        mac.stop()


def test_mac_defaults_avoid_window_manager_shortcuts(
    mac: MacHotkeyManager, carbon: FakeCarbon, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eye_tracker.config import LEGACY_HOTKEYS, HotkeySettings

    monkeypatch.setattr(sys, "platform", "darwin")
    control_option, command = 0x1000 | 0x0800, 0x0100
    # Rectangle and Magnet hold ⌃⌥T (Last Two Thirds) and ⌃⌥C (Center) by default.
    for key in ("t", "c"):
        carbon.reject[(hotkeys._MAC_KEYCODES[key], control_option)] = -9878
    defaults = HotkeySettings()
    for name in ("toggle_tracking", "toggle_privacy", "recalibrate"):
        assert mac.register(name, getattr(defaults, name), lambda: None), name
    assert {mods for _code, mods, _sig, _id in carbon.registered.values()} == {
        control_option | command
    }
    assert format_hotkey(hk(defaults.toggle_tracking)) == "⌃⌥⌘T"
    # The old defaults: registered exclusively, so the collision is reported clearly
    # instead of both apps reacting to one press.
    assert not mac.register("old", LEGACY_HOTKEYS[0], lambda: None)
    assert mac.last_error("old") == "⌃⌥T is already in use by another application"


def test_mac_layout_conflict_before_start_and_off_main_thread() -> None:
    e_code = hotkeys._MAC_KEYCODES["e"]
    carbon = FakeCarbon(MAC_US, modified={0x0800: {e_code: "´"}})
    mac = MacHotkeyManager(carbon=carbon)
    assert mac.layout_conflict("alt+e") is not None  # reads the layout on demand
    assert carbon.install_calls == 0
    result: list[str | None] = []
    worker = threading.Thread(target=lambda: result.append(mac.layout_conflict("alt+e")))
    worker.start()
    worker.join(2)
    assert result == [None]  # TIS calls are main-thread only: no answer, no crash


def test_mac_char_keymap_prefers_us_position_for_duplicates() -> None:
    chars = {0x0A: "<", 0x32: "<", 0x2B: ",", 0x31: " ", 0x24: "ab", 0x00: "Q"}
    keymap = hotkeys._mac_char_keymap(chars)
    assert keymap == {"<": 0x0A, ",": 0x2B, "q": 0x00}
    assert hotkeys._mac_char_keymap({0x05: "x", 0x07: "x"}) == {"x": 0x07}  # US 'x' is 0x07
    assert hotkeys._mac_keycode("f5", {"f": 0x03}) == 0x60  # non-character keys: fixed codes
    assert hotkeys._mac_keycode("minus", {}) is None
    assert hotkeys._mac_keycode("minus", None) == 0x1B


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
        self.RemoveEventHandler = _Fn(self._remove_handler)
        self.RegisterEventHotKey = _Fn(self._register)
        self.UnregisterEventHotKey = _Fn(
            lambda ref: 0 if self.hotkeys.pop(_value(ref), None) else -50
        )
        self.GetEventParameter = _Fn(self._get_parameter)

    def _remove_handler(self, ref: Any) -> int:
        self.removed.append(_value(ref))
        return 0

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
        assert options == 1  # kEventHotKeyExclusive: other apps' combos are reported
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


class FakeLayoutLib(FakeCarbonLib):
    """:class:`FakeCarbonLib` plus Text Input Sources, UCKeyTranslate and the
    distributed notification center (one object stands in for every framework)."""

    CURRENT, ASCII = 0x7001, 0x7002
    LAYOUT_KEY, CHANGED = 0xA1, 0xA2
    CENTER = 0xCE

    def __init__(self, current: dict[int, str], ascii_capable: dict[int, str]) -> None:
        super().__init__()
        self.sources = {self.CURRENT: current, self.ASCII: ascii_capable}
        self.option_layer: dict[int, str] = {}
        self.released: list[int] = []
        self.observers: list[tuple[Any, Any, Any]] = []
        self.removed_observers: list[Any] = []
        self.TISCopyCurrentKeyboardLayoutInputSource = _Fn(lambda: self.CURRENT)
        self.TISCopyCurrentASCIICapableKeyboardLayoutInputSource = _Fn(lambda: self.ASCII)
        self.TISGetInputSourceProperty = _Fn(
            lambda source, key: source + 0x1000 if key == self.LAYOUT_KEY else None
        )
        self.CFDataGetBytePtr = _Fn(lambda data: data + 0x1000)  # "layout" = source + 0x2000
        self.CFRelease = _Fn(self.released.append)
        self.LMGetKbdType = _Fn(lambda: 40)
        self.UCKeyTranslate = _Fn(self._translate)
        self.CFNotificationCenterGetDistributedCenter = _Fn(lambda: self.CENTER)
        self.CFNotificationCenterAddObserver = _Fn(self._add_observer)
        self.CFNotificationCenterRemoveObserver = _Fn(
            lambda center, observer, name, obj: self.removed_observers.append((center, name))
        )

    def _translate(
        self,
        layout: int,
        keycode: int,
        action: int,
        state: int,
        kbd_type: int,
        options: int,
        dead: Any,
        max_length: int,
        length: Any,
        buf: Any,
    ) -> int:
        assert (action, kbd_type, options, max_length) == (0, 40, 1, len(buf))
        source = layout - 0x2000
        chars = self.option_layer if state == 0x08 else self.sources[source]
        text = chars.get(keycode, "")
        for index, char in enumerate(text):
            buf[index] = ord(char)
        length._obj.value = len(text)
        return 0

    def _add_observer(
        self, center: int, observer: Any, proc: Any, name: int, obj: Any, behaviour: int
    ) -> None:
        assert (center, name, obj, behaviour) == (self.CENTER, self.CHANGED, None, 4)
        self.observers.append((observer, proc, name))


def test_mac_ctypes_layout_binding_with_fake_library(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    lib = FakeLayoutLib(MAC_AZERTY, MAC_AZERTY)
    lib.option_layer = {hotkeys._MAC_KEYCODES["e"]: "´"}
    monkeypatch.setattr(ctypes, "CDLL", lambda path: lib)
    constants = {
        "kTISPropertyUnicodeKeyLayoutData": FakeLayoutLib.LAYOUT_KEY,
        "kTISNotifySelectedKeyboardInputSourceChanged": FakeLayoutLib.CHANGED,
    }
    monkeypatch.setattr(hotkeys, "_cf_global", lambda _lib, name: constants[name])
    manager = MacHotkeyManager()
    try:
        assert manager.register("a", "ctrl+alt+a", lambda: None)
        ((code, _mods, _sig, _id),) = lib.hotkeys.values()
        assert code == hotkeys._MAC_KEYCODES["q"]  # AZERTY A, read through UCKeyTranslate
        assert manager.layout_conflict("alt+e") == "⌥E types '´' on the current keyboard layout"
        # Every copied input source is released again.
        assert lib.released
        assert set(lib.released) <= {FakeLayoutLib.CURRENT, FakeLayoutLib.ASCII}
        # The user switches to US: the distributed notification moves the hotkey.
        ((_observer, proc, _name),) = lib.observers
        lib.sources = {FakeLayoutLib.CURRENT: MAC_US, FakeLayoutLib.ASCII: MAC_US}
        proc(FakeLayoutLib.CENTER, None, FakeLayoutLib.CHANGED, None, None)
        ((code, _mods, _sig, _id),) = lib.hotkeys.values()
        assert code == hotkeys._MAC_KEYCODES["a"]
    finally:
        manager.stop()
    assert lib.removed_observers == [(FakeLayoutLib.CENTER, FakeLayoutLib.CHANGED)]


def test_mac_layout_api_missing_falls_back_to_us_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    lib = FakeCarbonLib()  # hot-key functions only: no TIS / UCKeyTranslate
    monkeypatch.setattr(ctypes, "CDLL", lambda path: lib)
    carbon = hotkeys._Carbon()
    assert carbon.layout_characters() is None
    assert carbon.observe_layout_changes(lambda: None) is False
    carbon.stop_observing_layout_changes()  # nothing to stop: no error
    assert carbon._layout_api is False  # the failed lookup is not repeated


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
        self.properties: dict[int, Any] = {}  # atom → property value
        self.property_reads = 0
        self.broken_reads = False
        self.event_mask: int | None = None  # what this client selected on the root

    def get_full_property(self, atom: int, property_type: int) -> Any:
        self.property_reads += 1
        if self.broken_reads:
            raise ConnectionError("lost the X server")
        value = self.properties.get(atom)
        return None if value is None else types.SimpleNamespace(value=value)

    def change_attributes(self, event_mask: int = 0, onerror: Any = None) -> None:
        self.event_mask = event_mask

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
        self.atoms: dict[str, int] = {}
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

    def intern_atom(self, name: str, only_if_exists: bool = False) -> int:
        if name not in self.atoms:
            if only_if_exists:
                return 0  # X.NONE
            self.atoms[name] = 100 + len(self.atoms)
        return self.atoms[name]

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

    def set_xkb_options(self, *options: str) -> int:
        """Record ``options`` in ``_XKB_RULES_NAMES`` as setxkbmap does; returns the atom."""
        atom = self.intern_atom("_XKB_RULES_NAMES")
        names = ["evdev", "pc105", "us,ru", ",", ",".join(options)]
        self.root.properties[atom] = "\0".join(names).encode() + b"\0"
        return atom


def property_event(atom: int) -> Any:
    return types.SimpleNamespace(type=hotkeys._X_PROPERTY_NOTIFY, atom=atom, state=0)


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


@pytest.mark.parametrize("release_state", [0, 8, 4, 12 | 2 | 16])
def test_x11_releasing_modifiers_first_rearms_the_hotkey(
    x11: tuple[X11HotkeyManager, FakeDisplay], release_state: int
) -> None:
    # A KeyRelease reports the modifiers held just before it: lifting Ctrl/Alt first
    # gives state 0 (or Mod1/Control only). The hotkey must fire on every press.
    manager, _display = x11
    fired: list[int] = []
    binding = bind_x11(manager, "t", "ctrl+alt+p", lambda: fired.append(1))
    assert manager._grab_binding(binding)
    code = keycode_of("p")
    for n in range(1, 6):
        manager._handle_event(key_event(hotkeys._X_KEY_PRESS, code, 12, 1000 * n))
        manager._handle_event(key_event(hotkeys._X_KEY_RELEASE, code, release_state, 1000 * n + 90))
    assert fired == [1] * 5


def test_x11_release_of_another_key_keeps_the_hotkey_held(
    x11: tuple[X11HotkeyManager, FakeDisplay],
) -> None:
    manager, _display = x11
    fired: list[str] = []
    p = bind_x11(manager, "p", "ctrl+alt+p", lambda: fired.append("p"))
    k = bind_x11(manager, "k", "ctrl+alt+shift+k", lambda: fired.append("k"))
    assert manager._grab_binding(p)
    assert manager._grab_binding(k)
    press, release = hotkeys._X_KEY_PRESS, hotkeys._X_KEY_RELEASE
    manager._handle_event(key_event(press, keycode_of("p"), 12, 100))
    manager._handle_event(key_event(release, keycode_of("k"), 0, 200))  # not P's release
    manager._handle_event(key_event(press, keycode_of("p"), 12, 300))  # P still held
    assert fired == ["p"]
    manager._handle_event(key_event(release, keycode_of("p"), 0, 400))
    manager._handle_event(key_event(press, keycode_of("k"), 13, 500))
    manager._handle_event(key_event(press, keycode_of("p"), 12, 600))
    assert fired == ["p", "k", "p"]


def test_x11_hotkey_lost_by_a_layout_change_comes_back(
    x11: tuple[X11HotkeyManager, FakeDisplay],
) -> None:
    manager, display = x11
    fired: list[int] = []
    binding = bind_x11(manager, "t", "ctrl+alt+p", lambda: fired.append(1))
    assert manager._grab_binding(binding)
    mapping = types.SimpleNamespace(type=hotkeys._X_MAPPING_NOTIFY, request=1)
    code = display.keymap.pop(KEYSYMS["p"])  # e.g. `setxkbmap ru`: no Latin keysyms
    manager._handle_event(mapping)
    assert display.root.grabs == set()
    assert manager.registered == {}  # not shown as working any more …
    error = manager.last_error("t")
    assert error is not None
    assert "not on the current keyboard layout" in error
    display.keymap[KEYSYMS["p"]] = code  # `setxkbmap us`
    manager._handle_event(mapping)  # … but retried on the next change
    assert (code, 12) in display.root.grabs
    assert manager.registered == {"t": parse_hotkey("ctrl+alt+p")}
    assert manager.last_error("t") is None
    manager._handle_event(key_event(hotkeys._X_KEY_PRESS, code, 12, 10))
    assert fired == [1]


def test_x11_regrab_keeps_in_flight_and_drops_removed_bindings(
    x11: tuple[X11HotkeyManager, FakeDisplay],
) -> None:
    manager, display = x11
    # Registration in flight: published in _by_id and grabbed on the hotkey thread,
    # but register() has not added it to _bindings yet.
    in_flight = hotkeys._Binding(manager._allocate_id(), "new", hk("ctrl+alt+k"), lambda: None)
    manager._by_id[in_flight.id] = in_flight
    assert manager._grab_binding(in_flight)
    # Removal in flight: _remove() forgot the binding, its ungrab is still queued.
    removed = hotkeys._Binding(manager._allocate_id(), "old", hk("ctrl+alt+j"), lambda: None)
    assert manager._grab_binding(removed)
    manager._handle_event(types.SimpleNamespace(type=hotkeys._X_MAPPING_NOTIFY, request=0))
    assert {code for code, _mods in display.root.grabs} == {keycode_of("k")}
    assert set(manager._grabs) == {in_flight.id}


def test_x11_loop_waits_without_timeout(
    x11: tuple[X11HotkeyManager, FakeDisplay], monkeypatch: pytest.MonkeyPatch
) -> None:
    import select

    manager, _display = x11
    timeouts: list[float | None] = []

    def fake_select(read: Any, write: Any, error: Any, timeout: float | None = None) -> Any:
        timeouts.append(timeout)
        raise OSError("end of test")  # ends the loop

    monkeypatch.setattr(select, "select", fake_select)
    manager._thread_loop()
    # No periodic wake-ups while idle: only events and the wake pipe end the wait.
    assert timeouts == [None]


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


# -------------------------------------------------- X11 keyboard-layout switches (XKB)
ALT_SHIFT = "grp:alt_shift_toggle"
SWITCHES_LAYOUT = "which switches the keyboard layout"


@pytest.fixture
def x11_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Label the meta key "Super", as on the systems that run X11."""
    monkeypatch.setattr(sys, "platform", "linux")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            b"evdev\0pc105\0us,ru\0,\0grp:alt_shift_toggle,terminate:ctrl_alt_bksp\0",
            (ALT_SHIFT, "terminate:ctrl_alt_bksp"),
        ),
        ("evdev\0pc105\0us\0\0grp:ctrl_shift_toggle", ("grp:ctrl_shift_toggle",)),
        (array.array("B", b"evdev\0pc105\0us\0\0 grp:lwin_toggle , \0"), ("grp:lwin_toggle",)),
        (bytearray(b"base\0pc105\0us\0\0\0"), ()),
        (b"evdev\0pc105\0us", ()),  # truncated
        (b"", ()),
        (None, ()),  # no such property
        (12, ()),  # not a string at all
    ],
)
def test_x11_parse_xkb_options(value: Any, expected: tuple[str, ...]) -> None:
    assert hotkeys._x11_parse_xkb_options(value) == expected


@pytest.mark.parametrize(
    ("text", "options", "expected"),
    [
        # The finding: Ctrl+Alt+Shift (the old Linux default) with an Alt+Shift switch.
        (
            "ctrl+alt+shift+t",
            [ALT_SHIFT],
            f"Ctrl+Alt+Shift+T includes Alt+Shift, {SWITCHES_LAYOUT} (XKB option {ALT_SHIFT})",
        ),
        # The first option that takes keys of the hotkey is named.
        (
            "ctrl+alt+shift+p",
            ["terminate:ctrl_alt_bksp", "grp:ctrl_shift_toggle", ALT_SHIFT],
            f"Ctrl+Alt+Shift+P includes Ctrl+Shift, {SWITCHES_LAYOUT} "
            "(XKB option grp:ctrl_shift_toggle)",
        ),
        # Today's default with the rare switches that do take its keys.
        (
            "ctrl+alt+meta+c",
            ["grp:ctrl_alt_toggle"],
            f"Ctrl+Alt+Super+C includes Ctrl+Alt, {SWITCHES_LAYOUT} "
            "(XKB option grp:ctrl_alt_toggle)",
        ),
        (
            "ctrl+alt+meta+t",
            ["grp:lwin_toggle"],
            f"Ctrl+Alt+Super+T includes Super, {SWITCHES_LAYOUT} (XKB option grp:lwin_toggle)",
        ),
        (
            "ctrl+alt+meta+t",
            ["lv3:lwin_switch"],
            "Ctrl+Alt+Super+T includes Super, which acts as AltGr (XKB option lv3:lwin_switch)",
        ),
        # Key toggles only concern their key.
        (
            "ctrl+alt+space",
            ["grp:alt_space_toggle"],
            f"Ctrl+Alt+Space includes Alt+Space, {SWITCHES_LAYOUT} "
            "(XKB option grp:alt_space_toggle)",
        ),
        (
            "meta+space",
            ["grp:win_space_toggle"],
            "Super+Space switches the keyboard layout (XKB option grp:win_space_toggle)",
        ),
        ("ctrl+alt+t", ["grp:alt_space_toggle", "grp:win_space_toggle"], None),
        ("ctrl+shift+t", [ALT_SHIFT, "grp:nonexistent_option"], None),
        ("ctrl+alt+shift+t", [], None),
    ],
)
def test_x11_layout_switch_conflict(
    x11_labels: None, text: str, options: list[str], expected: str | None
) -> None:
    assert hotkeys._x11_layout_switch_conflict(hk(text), options) == expected


#: Layout switches and other XKB options people commonly set (GNOME Tweaks, KDE and
#: Xfce keyboard settings, setxkbmap guides), none of which takes keys of a default.
COMMON_XKB_OPTIONS = [
    ALT_SHIFT,
    "grp:alt_shift_toggle_bidir",
    "grp:lalt_lshift_toggle",
    "grp:ctrl_shift_toggle",
    "grp:lctrl_lshift_toggle",
    "grp:win_space_toggle",
    "grp:caps_toggle",
    "grp:shift_caps_toggle",
    "grp:alt_caps_toggle",
    "grp:shifts_toggle",
    "grp:alts_toggle",
    "grp:ctrls_toggle",
    "grp:toggle",
    "grp:menu_toggle",
    "grp:rwin_toggle",
    "grp:ralt_rshift_toggle",
    "grp:rctrl_ralt_toggle",
    "grp:sclk_toggle",
    "lv3:ralt_switch",
    "lv3:caps_switch",
    "compose:ralt",
    "compose:menu",
    "terminate:ctrl_alt_bksp",
    "grp_led:scroll",
    "ctrl:nocaps",
    "caps:escape",
]


def test_x11_default_hotkeys_survive_the_usual_layout_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eye_tracker.config import LEGACY_HOTKEYS_X11, HotkeySettings

    monkeypatch.setattr(sys, "platform", "linux")
    defaults = HotkeySettings()
    for text in (defaults.toggle_tracking, defaults.toggle_privacy, defaults.recalibrate):
        assert hotkeys._x11_layout_switch_conflict(hk(text), COMMON_XKB_OPTIONS) is None, text
    # The previous defaults could not be pressed with the two most common switches.
    for option in (ALT_SHIFT, "grp:ctrl_shift_toggle"):
        for text in LEGACY_HOTKEYS_X11:
            assert hotkeys._x11_layout_switch_conflict(hk(text), [option]) is not None


def test_x11_layout_switch_table_is_consistent() -> None:
    for option, taken in hotkeys._X_KEY_OPTIONS.items():
        assert option.startswith(("grp:", "lv3:")), option
        assert taken.modifiers, option
        assert taken.modifiers <= set(MODIFIERS), option
        assert taken.key is None or taken.key in KEYS, option
    # Right-hand-only switches leave the keys that chords are pressed with alone.
    for option in ("grp:ralt_rshift_toggle", "grp:rctrl_rshift_toggle", "grp:rwin_toggle"):
        assert option not in hotkeys._X_KEY_OPTIONS


def test_x11_refuses_a_hotkey_the_layout_switch_makes_unpressable(
    x11: tuple[X11HotkeyManager, FakeDisplay], x11_labels: None
) -> None:
    manager, display = x11
    display.set_xkb_options(ALT_SHIFT, "terminate:ctrl_alt_bksp")
    manager._follow_xkb_options()
    assert manager._xkb_options == (ALT_SHIFT, "terminate:ctrl_alt_bksp")
    assert display.root.event_mask == hotkeys._X_PROPERTY_CHANGE_MASK
    blocked = bind_x11(manager, "t", "ctrl+alt+shift+t", lambda: None)
    # XGrabKey would succeed and the hotkey would silently never fire: refuse it.
    assert not manager._grab_binding(blocked)
    assert display.root.grab_calls == []
    assert manager.last_error("t") == (
        f"Ctrl+Alt+Shift+T includes Alt+Shift, {SWITCHES_LAYOUT} (XKB option {ALT_SHIFT})"
    )
    assert manager.layout_conflict("ctrl+alt+shift+t") == manager.last_error("t")
    assert manager.layout_conflict(hk("ctrl+alt+meta+t")) is None
    assert not display.closed  # answered from what the hotkey thread knows
    fine = bind_x11(manager, "p", "ctrl+alt+meta+p", lambda: None)
    assert manager._grab_binding(fine)
    assert (keycode_of("p"), 4 | 8 | 64) in display.root.grabs


def test_x11_follows_layout_switch_changes(
    x11: tuple[X11HotkeyManager, FakeDisplay], x11_labels: None
) -> None:
    manager, display = x11
    manager._follow_xkb_options()  # no _XKB_RULES_NAMES yet: nothing to avoid
    assert manager._xkb_options == ()
    fired: list[int] = []
    binding = bind_x11(manager, "t", "ctrl+alt+shift+t", lambda: fired.append(1))
    assert manager._grab_binding(binding)
    code, mods = keycode_of("t"), 1 | 4 | 8
    # The desktop applies the user's Alt+Shift switch after the app started (login).
    atom = display.set_xkb_options(ALT_SHIFT)
    reads = display.root.property_reads
    manager._handle_event(property_event(atom + 1))  # another root property: ignored
    assert display.root.property_reads == reads
    assert (code, mods) in display.root.grabs
    manager._handle_event(property_event(atom))
    assert display.root.grabs == set()
    assert manager.registered == {}  # reported as not working …
    error = manager.last_error("t")
    assert error is not None
    assert ALT_SHIFT in error
    # … an unchanged value (desktops rewrite it) does not churn the grabs …
    grab_calls = len(display.root.grab_calls)
    manager._handle_event(property_event(atom))
    assert len(display.root.grab_calls) == grab_calls
    # … and the hotkey comes back when the switch moves to Super+Space.
    display.set_xkb_options("grp:win_space_toggle")
    manager._handle_event(property_event(atom))
    assert (code, mods) in display.root.grabs
    assert manager.registered == {"t": hk("ctrl+alt+shift+t")}
    assert manager.last_error("t") is None
    manager._handle_event(key_event(hotkeys._X_KEY_PRESS, code, mods, 10))
    assert fired == [1]


def test_x11_mapping_change_rereads_the_layout_switches(
    x11: tuple[X11HotkeyManager, FakeDisplay], x11_labels: None
) -> None:
    manager, display = x11
    manager._follow_xkb_options()
    binding = bind_x11(manager, "p", "ctrl+shift+p", lambda: None)
    assert manager._grab_binding(binding)
    display.set_xkb_options("grp:ctrl_shift_toggle")
    manager._handle_event(types.SimpleNamespace(type=hotkeys._X_MAPPING_NOTIFY, request=1))
    assert manager.registered == {}
    error = manager.last_error("p")
    assert error is not None
    assert "grp:ctrl_shift_toggle" in error


def test_x11_failed_option_reads_keep_what_was_known(
    x11: tuple[X11HotkeyManager, FakeDisplay],
) -> None:
    manager, display = x11
    atom = display.set_xkb_options(ALT_SHIFT)
    manager._follow_xkb_options()
    display.root.broken_reads = True
    manager._handle_event(property_event(atom))
    assert manager._xkb_options == (ALT_SHIFT,)  # no flip-flop on a failed read


def test_x11_hotkeys_work_when_the_options_cannot_be_followed(
    x11: tuple[X11HotkeyManager, FakeDisplay], monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, display = x11

    def broken(name: str, only_if_exists: bool = False) -> int:
        raise ConnectionError("lost the X server")

    monkeypatch.setattr(display, "intern_atom", broken)
    manager._follow_xkb_options()
    assert manager._rules_atom == 0
    assert manager._xkb_options == ()
    binding = bind_x11(manager, "t", "ctrl+alt+shift+t", lambda: None)
    assert manager._grab_binding(binding)
    manager._handle_event(property_event(0))  # nothing followed: ignored, no crash
    assert manager.registered == {"t": hk("ctrl+alt+shift+t")}


def test_x11_layout_conflict_before_the_hotkey_thread_runs(x11_labels: None) -> None:
    display = FakeDisplay()
    display.set_xkb_options(ALT_SHIFT)
    opened: list[str | None] = []

    def factory(name: str | None) -> FakeDisplay:
        opened.append(name)
        return display

    manager = X11HotkeyManager(":1", display_factory=factory)
    conflict = manager.layout_conflict("ctrl+alt+shift+c")
    assert conflict == (
        f"Ctrl+Alt+Shift+C includes Alt+Shift, {SWITCHES_LAYOUT} (XKB option {ALT_SHIFT})"
    )
    assert manager.layout_conflict("ctrl+alt+meta+c") is None
    # A short-lived, read-only connection per question: nothing grabbed or selected.
    assert opened == [":1", ":1"]
    assert display.closed
    assert display.root.grab_calls == []
    assert display.root.event_mask is None
    # A server without the property: nothing to check, no atom created.
    bare = FakeDisplay()
    assert (
        X11HotkeyManager(display_factory=lambda name: bare).layout_conflict("ctrl+alt+shift+c")
        is None
    )
    assert bare.atoms == {}

    def unreachable(name: str | None) -> Any:
        raise ConnectionError("no X server")

    # Advisory only: an unreachable server never breaks the caller.
    assert X11HotkeyManager(display_factory=unreachable).layout_conflict("ctrl+alt+shift+c") is None
    with pytest.raises(ValueError, match="needs a key"):
        manager.layout_conflict("ctrl+alt")


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
def test_x11_thread_refuses_hotkeys_a_layout_switch_takes(
    fake_keysyms: None, x11_labels: None
) -> None:
    display = FakeDisplay(selectable=True)
    atom = display.set_xkb_options(ALT_SHIFT)
    manager = X11HotkeyManager(display_factory=lambda name: display)
    try:
        assert not manager.register("t", "ctrl+alt+shift+t", lambda: None)
        error = manager.last_error("t")
        assert error is not None
        assert ALT_SHIFT in error
        assert display.root.grabs == set()
        assert manager.register("t", "ctrl+alt+meta+t", lambda: None)
        assert manager.layout_conflict("ctrl+alt+shift+t") is not None  # cached options
        assert display.root.event_mask == hotkeys._X_PROPERTY_CHANGE_MASK
        # The switch changes while the app runs: the thread follows it.
        assert manager.register("p", "ctrl+shift+p", lambda: None)
        display.set_xkb_options("grp:ctrl_shift_toggle")
        display.push(property_event(atom))
        # Each loop pass runs queued calls, then drains events: after two round
        # trips the pass that saw the pushed event has finished.
        for _ in range(2):
            assert manager._call_on_thread(lambda: True, False)
        assert set(manager.registered) == {"t"}
    finally:
        manager.stop()
    assert display.closed


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
        taken = "Ctrl+Alt+Shift+F24 is already in use by another application"
        assert second.last_error("t") == taken
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


@windows_only
def test_windows_altgr_collision_is_refused_before_registering() -> None:
    # A layout on which even AltGr+Shift+F24 would type: the manager must refuse it
    # without calling RegisterHotKey, so another owner can still take the combo.
    refusing = WindowsHotkeyManager(layout_probe=FakeLayoutProbe(everything="x"))
    other = WindowsHotkeyManager()
    try:
        assert not refusing.register("t", COMBO, lambda: None)
        assert refusing.registered == {}
        assert refusing.last_error("t") == (
            "Ctrl+Alt+Shift+F24 is AltGr+Shift+F24 and types 'x' on the US keyboard layout"
        )
        assert other.register("t", COMBO, lambda: None)  # nothing was registered
        # With Win held, AltGr is not involved: no check, registration proceeds.
        assert refusing.register("w", "ctrl+alt+meta+f22", lambda: None)
    finally:
        refusing.stop()
        other.stop()


@windows_only
def test_windows_punctuation_without_a_key_is_refused(
    win: WindowsHotkeyManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hotkeys, "_win_virtual_key", lambda *args: None)
    assert not win.register("t", "ctrl+alt+shift+/", lambda: None)
    assert win.last_error("t") == (
        "Ctrl+Alt+Shift+/: no key types '/' without Shift or AltGr on the current keyboard layout"
    )


@windows_only
def test_windows_live_layout_queries_are_read_only_and_sane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eye_tracker.config import HotkeySettings

    probe = hotkeys._Win32LayoutProbe(hotkeys._win32())
    installed = probe.layouts()
    assert installed  # every session has at least one keyboard layout
    for hkl in installed:
        assert probe.altgr_text(0x87, False, hkl) is None  # VK_F24 never types
        assert probe.layout_name(hkl)
    assert hotkeys._win_layout_text(US) == "US"
    assert hotkeys._win_layout_text(0xF0020409) == "United States-Dvorak"  # "Layout Id" 0002
    manager = WindowsHotkeyManager()
    monkeypatch.setattr(sys, "platform", "win32")
    defaults = HotkeySettings()
    for text in (defaults.toggle_tracking, defaults.toggle_privacy, defaults.recalibrate):
        assert manager.layout_conflict(text) is None
    if TURKISH_Q in installed:  # the maintainer's layout: Ctrl+Alt+T is AltGr+T = '₺'
        conflict = manager.layout_conflict("ctrl+alt+t")
        assert conflict is not None
        assert "'₺'" in conflict
