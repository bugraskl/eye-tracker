"""System-wide keyboard shortcuts.

The module has two halves:

* **Parsing and formatting** (:func:`parse_hotkey`, :func:`format_hotkey`) are pure
  Python and work everywhere. Hotkeys are stored in the settings file as short,
  case-insensitive strings such as ``"ctrl+alt+p"``.
* **Registration** is done by a :class:`HotkeyManager`. :func:`create_hotkey_manager`
  picks the native implementation for the running system:

  ===========  ==========================================================================
  Windows      ``RegisterHotKey`` on a dedicated message-loop thread
  macOS        Carbon ``RegisterEventHotKey`` (no permission prompt needed); the
               events are delivered by the Cocoa run loop that Qt already spins
  Linux/X11    passive key grabs through python-xlib on a dedicated thread
  Wayland      unsupported: bind ``eye-tracker ctl <command>`` in the desktop's own
               keyboard settings instead
  ===========  ==========================================================================

Callbacks may be invoked on *any* thread (the hotkey thread on Windows/X11, the
main thread on macOS). They must return quickly and marshal work to the Qt main
thread themselves, for example by emitting a queued signal.

Only the registered combinations are ever observed. Nothing here installs a
keyboard hook or sees any other key press.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import queue
import re
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

log = logging.getLogger(__name__)

HotkeyCallback = Callable[[], None]

MODIFIERS: tuple[str, ...] = ("ctrl", "alt", "shift", "meta")
"""Canonical modifier names. ``meta`` is the Windows key, Command on macOS and Super on Linux."""

_LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")
_DIGITS = tuple("0123456789")
_FKEYS = tuple(f"f{n}" for n in range(1, 25))
_NAMED_KEYS = (
    "space",
    "enter",
    "tab",
    "escape",
    "backspace",
    "insert",
    "delete",
    "home",
    "end",
    "pageup",
    "pagedown",
    "left",
    "right",
    "up",
    "down",
)
# Canonical punctuation names follow the X11 keysym names; the value is the character
# the key produces on a US layout.
_PUNCTUATION: dict[str, str] = {
    "minus": "-",
    "equal": "=",
    "bracketleft": "[",
    "bracketright": "]",
    "backslash": "\\",
    "semicolon": ";",
    "quote": "'",
    "grave": "`",
    "comma": ",",
    "period": ".",
    "slash": "/",
}

KEYS: tuple[str, ...] = _LETTERS + _DIGITS + _FKEYS + _NAMED_KEYS + tuple(_PUNCTUATION)
"""Every key name accepted by :class:`Hotkey` (canonical, lower case)."""

_KEY_SET = frozenset(KEYS)

# Keys that produce text or drive text editing. Grabbing them with Shift as the only
# modifier would make it impossible to type capitals or use Shift+Tab system-wide.
_TYPING_KEYS = frozenset(
    _LETTERS
    + _DIGITS
    + tuple(_PUNCTUATION)
    + ("space", "enter", "tab", "backspace", "delete", "insert")
)


def _invert(groups: dict[str, tuple[str, ...]]) -> dict[str, str]:
    return {alias: canonical for canonical, aliases in groups.items() for alias in aliases}


_MODIFIER_ALIASES: dict[str, str] = _invert(
    {
        "ctrl": ("ctrl", "control", "ctl", "strg", "⌃"),
        "alt": ("alt", "option", "opt", "⌥"),
        "shift": ("shift", "⇧"),
        "meta": ("meta", "cmd", "command", "win", "windows", "super", "⌘"),
    }
)
_MODIFIER_SYMBOLS = frozenset("⌃⌥⇧⌘")

_KEY_ALIASES: dict[str, str] = _invert(
    {
        # Qt portable names, common spellings and the macOS menu glyphs (so that
        # format_hotkey(..., macos=True) round-trips through parse_hotkey).
        "escape": ("esc", "⎋"),
        "enter": ("return", "↩", "⏎", "↵"),
        "tab": ("⇥",),
        "space": ("spacebar", "␣"),
        "backspace": ("bksp", "⌫"),
        "insert": ("ins", "help"),
        "delete": ("del", "forwarddelete", "⌦"),
        "home": ("↖",),
        "end": ("↘",),
        "pageup": ("pgup", "prior", "⇞"),
        "pagedown": ("pgdn", "pgdown", "next", "⇟"),
        "left": ("arrowleft", "leftarrow", "←"),
        "right": ("arrowright", "rightarrow", "→"),
        "up": ("arrowup", "uparrow", "↑"),
        "down": ("arrowdown", "downarrow", "↓"),
        "minus": ("dash", "hyphen"),
        "equal": ("equals",),
        "bracketleft": ("leftbracket", "lbracket"),
        "bracketright": ("rightbracket", "rbracket"),
        "quote": ("apostrophe",),
        "grave": ("backquote", "backtick"),
        "period": ("dot", "fullstop"),
        "slash": ("forwardslash",),
    }
)
_KEY_ALIASES.update({char: name for name, char in _PUNCTUATION.items()})

_DISPLAY_NAMES: dict[str, str] = {
    "space": "Space",
    "enter": "Enter",
    "tab": "Tab",
    "escape": "Esc",
    "backspace": "Backspace",
    "insert": "Insert",
    "delete": "Delete",
    "home": "Home",
    "end": "End",
    "pageup": "PgUp",
    "pagedown": "PgDn",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    **_PUNCTUATION,
}
_MAC_DISPLAY_NAMES: dict[str, str] = {
    **_DISPLAY_NAMES,
    "enter": "↩",
    "tab": "⇥",
    "escape": "⎋",
    "backspace": "⌫",
    "insert": "Help",
    "delete": "⌦",
    "home": "↖",
    "end": "↘",
    "pageup": "⇞",
    "pagedown": "⇟",
    "left": "←",
    "right": "→",
    "up": "↑",
    "down": "↓",
}
_MAC_MODIFIER_SYMBOLS = {"ctrl": "⌃", "alt": "⌥", "shift": "⇧", "meta": "⌘"}

_SEPARATORS = re.compile(r"[\s_\-]+")


# --------------------------------------------------------------------------- model
@dataclass(frozen=True)
class Hotkey:
    """A key plus at least one modifier.

    ``modifiers`` is a subset of :data:`MODIFIERS`; ``key`` is one of :data:`KEYS`.
    ``str(hotkey)`` gives the canonical settings form, e.g. ``"ctrl+alt+p"``.
    Construction validates both and raises :class:`ValueError` otherwise.
    """

    modifiers: frozenset[str]
    key: str

    def __post_init__(self) -> None:
        # Accept any iterable of modifier names but always store a frozenset so that
        # equality and hashing do not depend on how the object was built.
        mods = frozenset(self.modifiers)
        object.__setattr__(self, "modifiers", mods)
        unknown = sorted(mods.difference(MODIFIERS))
        if unknown:
            raise ValueError(f"unknown modifier(s): {', '.join(unknown)}")
        if self.key not in _KEY_SET:
            raise ValueError(f"unsupported key: {self.key!r}")
        if not mods:
            raise ValueError("a global hotkey needs at least one modifier (Ctrl, Alt, Shift, Meta)")
        if mods == {"shift"} and self.key in _TYPING_KEYS:
            raise ValueError("Shift alone is not enough for a typing key; add Ctrl, Alt or Meta")

    @property
    def ordered_modifiers(self) -> tuple[str, ...]:
        """Modifiers in canonical order (ctrl, alt, shift, meta)."""
        return tuple(m for m in MODIFIERS if m in self.modifiers)

    def __str__(self) -> str:
        return "+".join((*self.ordered_modifiers, self.key))


def _normalise_token(token: str) -> str:
    token = token.strip().lower()
    # Single characters are keys in their own right ("-" is the minus key), so only
    # multi-character words have their inner separators removed ("Page Up" → "pageup").
    if len(token) > 1:
        token = _SEPARATORS.sub("", token)
    return token


def _split_tokens(text: str) -> list[str]:
    """Split ``"Ctrl + Alt+P"`` or ``"⌃⌥P"`` into raw tokens (last one is the key)."""
    tokens: list[str] = []
    for raw in text.split("+"):
        part = raw.strip()
        peeled = False
        # macOS style glyphs may be glued to each other and to the key: "⌃⌥⇧P".
        while len(part) > 1 and part[0] in _MODIFIER_SYMBOLS:
            tokens.append(part[0])
            part = part[1:].strip()
            peeled = True
        if part or not peeled:
            tokens.append(part)
    return tokens


def parse_hotkey(text: str) -> Hotkey:
    """Parse a user-facing hotkey string.

    Accepts ``"Ctrl+Alt+P"``, ``"ctrl + alt + p"``, ``"cmd+shift+g"`` and macOS glyph
    notation such as ``"⌃⌥P"``. Matching is case-insensitive and tolerant of spaces.
    Aliases: ``cmd``/``command``/``win``/``windows``/``super`` → ``meta``,
    ``option``/``opt`` → ``alt``, ``control``/``ctl`` → ``ctrl``; key aliases include
    ``esc``, ``return``, ``del``, ``ins``, ``pgup``/``pgdn`` and punctuation characters.

    Note for Qt callers: ``QKeySequence.toString(PortableText)`` on macOS calls the
    Command key ``Ctrl`` and the Control key ``Meta``; swap them before parsing.

    Raises:
        ValueError: if the text is empty, uses an unknown modifier or key, has no
            modifier, or consists of modifiers only.
    """
    if not isinstance(text, str):
        raise ValueError(f"hotkey must be a string, not {type(text).__name__}")
    stripped = text.strip()
    if not stripped:
        raise ValueError("hotkey is empty")
    if stripped == "+" or (stripped.endswith("+") and stripped[:-1].rstrip().endswith("+")):
        # "Ctrl++": '+' is Shift+'=' on most layouts, so it has no key of its own.
        raise ValueError(f"the '+' key cannot be used in hotkey {text!r}; use '=' instead")
    tokens = _split_tokens(stripped)
    *mod_tokens, key_token = tokens

    modifiers: set[str] = set()
    for raw in mod_tokens:
        token = _normalise_token(raw)
        if not token:
            raise ValueError(f"malformed hotkey {text!r}: empty part")
        modifier = _MODIFIER_ALIASES.get(token)
        if modifier is None:
            raise ValueError(f"unknown modifier {raw.strip()!r} in hotkey {text!r}")
        modifiers.add(modifier)

    key = _normalise_token(key_token)
    if not key:
        raise ValueError(f"malformed hotkey {text!r}: missing key")
    if key in _MODIFIER_ALIASES:
        raise ValueError(f"hotkey {text!r} needs a key in addition to its modifiers")
    key = _KEY_ALIASES.get(key, key)
    if key not in _KEY_SET:
        raise ValueError(f"unsupported key {key_token.strip()!r} in hotkey {text!r}")
    if not modifiers:
        raise ValueError(f"hotkey {text!r} needs at least one modifier (Ctrl, Alt, Shift or Meta)")
    return Hotkey(frozenset(modifiers), key)


def format_hotkey(hk: Hotkey, macos: bool | None = None) -> str:
    """Human-readable label: ``"Ctrl+Alt+P"`` (Windows/Linux) or ``"⌃⌥P"`` (macOS).

    ``macos=None`` follows the running system. The meta key is shown as ``Win`` on
    Windows and ``Super`` elsewhere. The result is always accepted by
    :func:`parse_hotkey`.
    """
    if macos is None:
        macos = sys.platform == "darwin"
    if macos:
        key = _MAC_DISPLAY_NAMES.get(hk.key, hk.key.upper())
        return "".join(_MAC_MODIFIER_SYMBOLS[m] for m in hk.ordered_modifiers) + key
    meta = "Win" if sys.platform == "win32" else "Super"
    labels = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "meta": meta}
    key = _DISPLAY_NAMES.get(hk.key, hk.key.upper())
    return "+".join([*(labels[m] for m in hk.ordered_modifiers), key])


# ------------------------------------------------------------------------- managers
class HotkeyManager:
    """Registers global hotkeys. This base class is the *unsupported* no-op.

    Native implementations share these semantics:

    * :meth:`register` starts the backend on demand and returns whether the OS
      accepted the combination (``False`` if another application owns it).
      Re-registering a name replaces its previous hotkey; registering a
      combination that another name in this manager already uses returns ``False``.
    * :meth:`start` is optional (it pre-warms the backend). :meth:`stop` releases
      every hotkey, ends the backend and is idempotent; a later :meth:`register`
      starts it again.
    * Callbacks run on an arbitrary thread; exceptions they raise are logged.
    """

    supported: ClassVar[bool] = False
    name: ClassVar[str] = "none"

    def __init__(self, note: str | None = None) -> None:
        #: Human-readable remark on availability (why hotkeys are unavailable or
        #: limited here); ``None`` when fully supported. Shown by the UI and doctor.
        self.note = note

    def register(self, name: str, hotkey: Hotkey | str, callback: HotkeyCallback) -> bool:
        """Bind ``hotkey`` to ``callback`` under ``name``. Returns success."""
        log.debug("Global hotkeys unsupported here; %r not registered", name)
        return False

    def unregister(self, name: str) -> bool:
        """Release the hotkey registered under ``name``. Returns whether one existed."""
        return False

    def unregister_all(self) -> None:
        """Release every hotkey (the backend keeps running)."""

    @property
    def registered(self) -> dict[str, Hotkey]:
        """Currently active registrations by name."""
        return {}

    def start(self) -> None:
        """Start the backend early (optional; :meth:`register` starts it on demand)."""

    def stop(self) -> None:
        """Release every hotkey and stop the backend. Safe to call repeatedly."""


@dataclass(eq=False)
class _Binding:
    id: int
    name: str
    hotkey: Hotkey
    callback: HotkeyCallback


class _NativeHotkeyManager(HotkeyManager):
    """Bookkeeping shared by the native managers.

    Subclasses implement the ``_backend_*`` and ``_native_*`` hooks. Two locks keep
    dispatch deadlock-free: ``_api_lock`` serialises public calls (and may be held
    while waiting for a backend thread), ``_table_lock`` only ever guards short
    dictionary operations and is the only lock taken when a hotkey fires.
    """

    supported: ClassVar[bool] = True
    _MAX_ID: ClassVar[int] = 0xBFFF  # RegisterHotKey accepts application ids 0..0xBFFF

    def __init__(self, note: str | None = None) -> None:
        super().__init__(note)
        self._api_lock = threading.RLock()
        self._table_lock = threading.Lock()
        self._bindings: dict[str, _Binding] = {}
        self._by_id: dict[int, _Binding] = {}
        self._last_id = 0

    # ------------------------------------------------------------------ public
    def register(self, name: str, hotkey: Hotkey | str, callback: HotkeyCallback) -> bool:
        if not callable(callback):
            raise TypeError("callback must be callable")
        if isinstance(hotkey, str):
            try:
                hotkey = parse_hotkey(hotkey)
            except ValueError as exc:
                log.warning("Hotkey for %r not registered: %s", name, exc)
                return False
        label = format_hotkey(hotkey)
        with self._api_lock:
            existing = self._bindings.get(name)
            if existing is not None and existing.hotkey == hotkey:
                with self._table_lock:
                    existing.callback = callback
                return True
            clash = next(
                (b for b in self._bindings.values() if b.hotkey == hotkey and b.name != name),
                None,
            )
            if clash is not None:
                log.warning("Hotkey %s for %r is already bound to %r", label, name, clash.name)
                return False
            if existing is not None:
                self._remove(existing)
            if not self._backend_start():
                log.warning("Hotkey %s for %r not registered: backend unavailable", label, name)
                return False
            binding = _Binding(self._allocate_id(), name, hotkey, callback)
            # Publish before the OS registration so a press that arrives immediately
            # afterwards can already be dispatched.
            with self._table_lock:
                self._by_id[binding.id] = binding
            ok = False
            try:
                ok = bool(self._native_register(binding))
            except Exception:
                log.exception("Registering hotkey %s for %r failed", label, name)
            with self._table_lock:
                if ok:
                    self._bindings[name] = binding
                else:
                    self._by_id.pop(binding.id, None)
            if ok:
                log.info("Registered hotkey %s for %s", label, name)
            return ok

    def unregister(self, name: str) -> bool:
        with self._api_lock:
            binding = self._bindings.get(name)
            if binding is None:
                return False
            self._remove(binding)
            return True

    def unregister_all(self) -> None:
        with self._api_lock:
            for binding in list(self._bindings.values()):
                self._remove(binding)

    @property
    def registered(self) -> dict[str, Hotkey]:
        with self._table_lock:
            return {name: b.hotkey for name, b in self._bindings.items()}

    def start(self) -> None:
        with self._api_lock:
            if not self._backend_start():
                log.warning("Global hotkey backend %s could not start", self.name)

    def stop(self) -> None:
        with self._api_lock:
            for binding in list(self._bindings.values()):
                self._remove(binding)
            try:
                self._backend_stop()
            except Exception:
                log.exception("Stopping the hotkey backend failed")

    # --------------------------------------------------------------- internals
    def _allocate_id(self) -> int:
        with self._table_lock:
            for _ in range(self._MAX_ID):
                self._last_id = self._last_id % self._MAX_ID + 1
                if self._last_id not in self._by_id:
                    return self._last_id
        raise RuntimeError("no free hotkey id")  # pragma: no cover - 49k live hotkeys

    def _remove(self, binding: _Binding) -> None:
        # Forget the binding first so that a press racing with the unregistration is
        # simply ignored instead of calling a callback the owner already dropped.
        with self._table_lock:
            self._bindings.pop(binding.name, None)
            self._by_id.pop(binding.id, None)
        try:
            self._native_unregister(binding)
        except Exception:
            log.exception("Unregistering hotkey %s failed", format_hotkey(binding.hotkey))

    def _dispatch(self, binding_id: int) -> None:
        with self._table_lock:
            binding = self._by_id.get(binding_id)
            callback = binding.callback if binding is not None else None
        if binding is None or callback is None:
            log.debug("Ignoring event for unknown hotkey id %s", binding_id)
            return
        log.debug("Hotkey %s pressed (%s)", format_hotkey(binding.hotkey), binding.name)
        try:
            callback()
        except Exception:
            log.exception("Hotkey callback for %r raised", binding.name)

    # ------------------------------------------------------------------- hooks
    def _backend_start(self) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def _backend_stop(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _native_register(self, binding: _Binding) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def _native_unregister(self, binding: _Binding) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class _PendingCall:
    """A function to run on the backend thread, with a result the caller waits for."""

    __slots__ = ("_lock", "_state", "done", "error", "fn", "result")

    def __init__(self, fn: Callable[[], Any]) -> None:
        self.fn = fn
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None
        self._lock = threading.Lock()
        self._state = "pending"  # → "running" → "finished", or "cancelled"

    def run(self) -> None:
        with self._lock:
            if self._state != "pending":
                return
            self._state = "running"
        try:
            self.result = self.fn()
        except Exception as exc:
            self.error = exc
        finally:
            with self._lock:
                self._state = "finished"
            self.done.set()

    def cancel(self) -> bool:
        """Prevent a call that has not started yet. Returns ``True`` if it was prevented."""
        with self._lock:
            if self._state != "pending":
                return False
            self._state = "cancelled"
        self.done.set()
        return True

    @property
    def finished(self) -> bool:
        return self._state == "finished"


class _ThreadedHotkeyManager(_NativeHotkeyManager):
    """A native manager whose OS registrations live on one dedicated thread.

    Windows hotkeys are bound to the registering thread's message queue and an
    Xlib ``Display`` must not be shared between threads, so every OS call is
    executed on the backend thread via :meth:`_call_on_thread`.
    """

    _CALL_TIMEOUT_S: ClassVar[float] = 2.0
    _JOIN_TIMEOUT_S: ClassVar[float] = 2.0

    def __init__(self, note: str | None = None) -> None:
        super().__init__(note)
        self._thread: threading.Thread | None = None
        self._calls: queue.SimpleQueue[_PendingCall] = queue.SimpleQueue()
        self._stopping = threading.Event()
        self._ready = threading.Event()
        self._start_ok = False

    # ------------------------------------------------------------- lifecycle
    def _backend_start(self) -> bool:
        thread = self._thread
        if thread is not None and thread.is_alive():
            if not self._stopping.is_set():
                return self._start_ok
            # A stop() issued from inside a callback leaves the thread finishing its
            # current message; never run two backend threads at once.
            if thread is threading.current_thread():
                return False
            thread.join(self._JOIN_TIMEOUT_S)
            if thread.is_alive():
                log.warning("Previous hotkey thread is still running")
                return False
        self._stopping.clear()
        self._ready.clear()
        self._start_ok = False
        if not self._before_thread_start():
            return False
        thread = threading.Thread(
            target=self._thread_main, name=f"eye-tracker-hotkeys-{self.name}", daemon=True
        )
        self._thread = thread
        thread.start()
        if not self._ready.wait(self._CALL_TIMEOUT_S):
            log.warning("Hotkey thread did not become ready in time")
            return False
        return self._start_ok

    def _backend_stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stopping.set()
        self._wake_for_stop()
        if thread is not threading.current_thread():
            thread.join(self._JOIN_TIMEOUT_S)
            if thread.is_alive():
                log.warning("Hotkey thread did not stop within %.1f s", self._JOIN_TIMEOUT_S)
        if not thread.is_alive():
            self._thread = None

    def _thread_main(self) -> None:
        try:
            self._start_ok = bool(self._thread_setup())
        except Exception:
            log.exception("Hotkey backend %s failed to initialise", self.name)
            self._start_ok = False
        self._ready.set()
        try:
            if self._start_ok:
                self._thread_loop()
        except Exception:
            log.exception("Hotkey thread (%s) crashed", self.name)
        finally:
            try:
                self._thread_teardown()
            except Exception:
                log.exception("Hotkey backend %s cleanup failed", self.name)
            self._cancel_pending_calls()

    # ------------------------------------------------------------ cross-thread
    def _call_on_thread(self, fn: Callable[[], Any], default: Any) -> Any:
        thread = self._thread
        if thread is None or not thread.is_alive():
            return default
        if threading.current_thread() is thread:
            return fn()
        if self._stopping.is_set():
            return default
        call = _PendingCall(fn)
        self._calls.put(call)
        if not self._wake():
            call.cancel()
            return default
        if not call.done.wait(self._CALL_TIMEOUT_S) and not call.cancel():
            # Already running: give the native call a little longer to finish.
            call.done.wait(self._CALL_TIMEOUT_S)
        if not call.finished:
            log.warning("Hotkey thread did not answer in time")
            return default
        if call.error is not None:
            log.error("Hotkey operation failed on the hotkey thread", exc_info=call.error)
            return default
        return call.result

    def _run_pending_calls(self) -> None:
        while True:
            try:
                call = self._calls.get_nowait()
            except queue.Empty:
                return
            call.run()

    def _cancel_pending_calls(self) -> None:
        while True:
            try:
                call = self._calls.get_nowait()
            except queue.Empty:
                return
            call.cancel()

    # ------------------------------------------------------------------- hooks
    def _before_thread_start(self) -> bool:
        """Prepare resources the thread needs (runs on the caller's thread)."""
        return True

    def _thread_setup(self) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def _thread_loop(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _thread_teardown(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _wake(self) -> bool:  # pragma: no cover - abstract
        """Make the thread run :meth:`_run_pending_calls` soon. Returns success."""
        raise NotImplementedError

    def _wake_for_stop(self) -> None:
        self._wake()


# ------------------------------------------------------------------------- Windows
_MOD_ALT = 0x0001
_MOD_CONTROL = 0x0002
_MOD_SHIFT = 0x0004
_MOD_WIN = 0x0008
_MOD_NOREPEAT = 0x4000
_WM_QUIT = 0x0012
_WM_HOTKEY = 0x0312
_WM_USER = 0x0400
_WM_APP_CALL = 0x8000 + 0x2E7  # WM_APP + private offset
_PM_NOREMOVE = 0x0000
_ERROR_HOTKEY_ALREADY_REGISTERED = 1409

_WIN_MODIFIERS = {"ctrl": _MOD_CONTROL, "alt": _MOD_ALT, "shift": _MOD_SHIFT, "meta": _MOD_WIN}

_WIN_VK: dict[str, int] = {
    **{ch: ord(ch.upper()) for ch in _LETTERS},
    **{d: ord(d) for d in _DIGITS},
    **{f"f{n}": 0x6F + n for n in range(1, 25)},  # VK_F1 = 0x70 … VK_F24 = 0x87
    "space": 0x20,
    "enter": 0x0D,
    "tab": 0x09,
    "escape": 0x1B,
    "backspace": 0x08,
    "insert": 0x2D,
    "delete": 0x2E,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    # OEM keys, US layout; other layouts are resolved with VkKeyScanW first.
    "semicolon": 0xBA,
    "equal": 0xBB,
    "comma": 0xBC,
    "minus": 0xBD,
    "period": 0xBE,
    "slash": 0xBF,
    "grave": 0xC0,
    "bracketleft": 0xDB,
    "backslash": 0xDC,
    "bracketright": 0xDD,
    "quote": 0xDE,
}


def _win_modifiers(modifiers: Iterable[str]) -> int:
    mask = 0
    for m in modifiers:
        mask |= _WIN_MODIFIERS[m]
    return mask


def _win_virtual_key(key: str, vk_scan: Callable[[str], int] | None = None) -> int | None:
    """Virtual-key code for ``key``.

    Punctuation lives on different keys in different layouts (on a Turkish Q
    keyboard "." is where "/" is on a US one), so the active layout is asked first
    via ``VkKeyScanW``; a mapping that needs Shift/AltGr is ignored because the user
    asked for the unshifted key.
    """
    char = _PUNCTUATION.get(key)
    if char is not None and vk_scan is not None:
        try:
            result = vk_scan(char)
        except Exception:
            result = -1
        if result != -1 and (result >> 8) & 0xFF == 0:
            return result & 0xFF
    return _WIN_VK.get(key)


class _Win32Api:
    """Private ctypes prototypes (own WinDLL instances so other modules' argtypes are
    never clobbered)."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.MSG = wintypes.MSG
        self.byref = ctypes.byref
        self.get_last_error = ctypes.get_last_error  # type: ignore[attr-defined]
        user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

        self.RegisterHotKey = user32.RegisterHotKey
        self.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        self.RegisterHotKey.restype = wintypes.BOOL
        self.UnregisterHotKey = user32.UnregisterHotKey
        self.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        self.UnregisterHotKey.restype = wintypes.BOOL
        self.GetMessageW = user32.GetMessageW
        self.GetMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self.GetMessageW.restype = wintypes.BOOL
        self.PeekMessageW = user32.PeekMessageW
        self.PeekMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self.PeekMessageW.restype = wintypes.BOOL
        self.PostThreadMessageW = user32.PostThreadMessageW
        self.PostThreadMessageW.argtypes = [
            wintypes.DWORD,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self.PostThreadMessageW.restype = wintypes.BOOL
        self.VkKeyScanW = user32.VkKeyScanW
        self.VkKeyScanW.argtypes = [wintypes.WCHAR]
        self.VkKeyScanW.restype = ctypes.c_short
        self.GetCurrentThreadId = kernel32.GetCurrentThreadId
        self.GetCurrentThreadId.argtypes = []
        self.GetCurrentThreadId.restype = wintypes.DWORD


@functools.cache
def _win32() -> _Win32Api:
    return _Win32Api()


class WindowsHotkeyManager(_ThreadedHotkeyManager):
    """``RegisterHotKey`` with ``MOD_NOREPEAT`` on a dedicated message-loop thread.

    Hotkeys registered with a ``NULL`` window belong to the registering thread, so
    registrations are posted to that thread (``PostThreadMessageW`` with a private
    ``WM_APP`` message) and executed there. ``GetMessageW`` blocks, so the thread
    costs nothing while idle.
    """

    name: ClassVar[str] = "win32"

    def __init__(self) -> None:
        super().__init__()
        self._thread_id = 0
        self._os_ids: set[int] = set()  # touched only on the hotkey thread

    def _thread_setup(self) -> bool:
        api = _win32()
        self._thread_id = int(api.GetCurrentThreadId())
        # PostThreadMessageW fails until the thread owns a message queue; any
        # Peek/GetMessage call creates it, so do that before announcing readiness.
        msg = api.MSG()
        api.PeekMessageW(api.byref(msg), None, _WM_USER, _WM_USER, _PM_NOREMOVE)
        return True

    def _thread_loop(self) -> None:
        api = _win32()
        msg = api.MSG()
        while not self._stopping.is_set():
            result = api.GetMessageW(api.byref(msg), None, 0, 0)
            if result == 0:  # WM_QUIT
                break
            if result == -1:
                log.error("GetMessageW failed (error %s)", api.get_last_error())
                break
            if msg.message == _WM_HOTKEY:
                self._dispatch(int(msg.wParam))
            elif msg.message == _WM_APP_CALL:
                self._run_pending_calls()

    def _thread_teardown(self) -> None:
        api = _win32()
        for hotkey_id in list(self._os_ids):
            api.UnregisterHotKey(None, hotkey_id)
        self._os_ids.clear()
        # Thread ids are recycled; never post to this one again.
        self._thread_id = 0

    def _post(self, message: int) -> bool:
        if not self._thread_id:
            return False
        return bool(_win32().PostThreadMessageW(self._thread_id, message, 0, 0))

    def _wake(self) -> bool:
        return self._post(_WM_APP_CALL)

    def _wake_for_stop(self) -> None:
        self._post(_WM_QUIT)

    def _native_register(self, binding: _Binding) -> bool:
        return bool(self._call_on_thread(lambda: self._register_on_thread(binding), False))

    def _native_unregister(self, binding: _Binding) -> None:
        self._call_on_thread(lambda: self._unregister_on_thread(binding.id), None)

    def _register_on_thread(self, binding: _Binding) -> bool:
        api = _win32()
        vk = _win_virtual_key(binding.hotkey.key, api.VkKeyScanW)
        if vk is None:
            log.warning("No virtual-key code for %r", binding.hotkey.key)
            return False
        mods = _win_modifiers(binding.hotkey.modifiers) | _MOD_NOREPEAT
        if not api.RegisterHotKey(None, binding.id, mods, vk):
            error = api.get_last_error()
            label = format_hotkey(binding.hotkey)
            if error == _ERROR_HOTKEY_ALREADY_REGISTERED:
                log.warning("Hotkey %s is already in use by another application", label)
            else:
                log.warning("RegisterHotKey(%s) failed (error %s)", label, error)
            return False
        self._os_ids.add(binding.id)
        return True

    def _unregister_on_thread(self, hotkey_id: int) -> None:
        if hotkey_id in self._os_ids:
            _win32().UnregisterHotKey(None, hotkey_id)
            self._os_ids.discard(hotkey_id)


# --------------------------------------------------------------------------- macOS
_MAC_CMD_KEY = 0x0100
_MAC_SHIFT_KEY = 0x0200
_MAC_OPTION_KEY = 0x0800
_MAC_CONTROL_KEY = 0x1000
_MAC_MODIFIERS = {
    "ctrl": _MAC_CONTROL_KEY,
    "alt": _MAC_OPTION_KEY,
    "shift": _MAC_SHIFT_KEY,
    "meta": _MAC_CMD_KEY,
}
_NO_ERR = 0
_EVENT_NOT_HANDLED_ERR = -9874
_EVENT_HOTKEY_EXISTS_ERR = -9878
_EVENT_HOTKEY_PRESSED = 5


def _fourcc(code: str) -> int:
    """Four-character Carbon ``OSType`` as an integer (``'keyb'`` → 0x6B657962)."""
    raw = code.encode("mac_roman")
    if len(raw) != 4:
        raise ValueError(f"four-char code must have 4 bytes: {code!r}")
    return int.from_bytes(raw, "big")


_EVENT_CLASS_KEYBOARD = _fourcc("keyb")
_EVENT_PARAM_DIRECT_OBJECT = _fourcc("----")
_TYPE_EVENT_HOTKEY_ID = _fourcc("hkid")
_HOTKEY_SIGNATURE = _fourcc("EyTk")

# kVK_* virtual key codes from HIToolbox/Events.h. They identify physical key
# positions on an ANSI (US) keyboard; macOS has no F21–F24.
_MAC_KEYCODES: dict[str, int] = {
    "a": 0x00,
    "s": 0x01,
    "d": 0x02,
    "f": 0x03,
    "h": 0x04,
    "g": 0x05,
    "z": 0x06,
    "x": 0x07,
    "c": 0x08,
    "v": 0x09,
    "b": 0x0B,
    "q": 0x0C,
    "w": 0x0D,
    "e": 0x0E,
    "r": 0x0F,
    "y": 0x10,
    "t": 0x11,
    "1": 0x12,
    "2": 0x13,
    "3": 0x14,
    "4": 0x15,
    "6": 0x16,
    "5": 0x17,
    "equal": 0x18,
    "9": 0x19,
    "7": 0x1A,
    "minus": 0x1B,
    "8": 0x1C,
    "0": 0x1D,
    "bracketright": 0x1E,
    "o": 0x1F,
    "u": 0x20,
    "bracketleft": 0x21,
    "i": 0x22,
    "p": 0x23,
    "enter": 0x24,
    "l": 0x25,
    "j": 0x26,
    "quote": 0x27,
    "k": 0x28,
    "semicolon": 0x29,
    "backslash": 0x2A,
    "comma": 0x2B,
    "slash": 0x2C,
    "n": 0x2D,
    "m": 0x2E,
    "period": 0x2F,
    "tab": 0x30,
    "space": 0x31,
    "grave": 0x32,
    "backspace": 0x33,  # kVK_Delete
    "escape": 0x35,
    "f17": 0x40,
    "f18": 0x4F,
    "f19": 0x50,
    "f20": 0x5A,
    "f5": 0x60,
    "f6": 0x61,
    "f7": 0x62,
    "f3": 0x63,
    "f8": 0x64,
    "f9": 0x65,
    "f11": 0x67,
    "f13": 0x69,
    "f16": 0x6A,
    "f14": 0x6B,
    "f10": 0x6D,
    "f12": 0x6F,
    "f15": 0x71,
    "insert": 0x72,  # kVK_Help
    "home": 0x73,
    "pageup": 0x74,
    "delete": 0x75,  # kVK_ForwardDelete
    "f4": 0x76,
    "end": 0x77,
    "f2": 0x78,
    "pagedown": 0x79,
    "f1": 0x7A,
    "left": 0x7B,
    "right": 0x7C,
    "down": 0x7D,
    "up": 0x7E,
}


def _mac_modifiers(modifiers: Iterable[str]) -> int:
    mask = 0
    for m in modifiers:
        mask |= _MAC_MODIFIERS[m]
    return mask


class _Carbon:
    """ctypes binding of the handful of Carbon (HIToolbox) hot-key functions.

    Every method returns plain Python values so :class:`MacHotkeyManager` can be
    exercised with a fake on any OS.
    """

    PATH = "/System/Library/Frameworks/Carbon.framework/Carbon"

    def __init__(self) -> None:
        import ctypes

        self._ctypes = ctypes
        lib = ctypes.CDLL(self.PATH)
        u32 = ctypes.c_uint32
        vp = ctypes.c_void_p

        class EventTypeSpec(ctypes.Structure):
            _fields_: ClassVar = [("eventClass", u32), ("eventKind", u32)]

        class EventHotKeyID(ctypes.Structure):
            _fields_: ClassVar = [("signature", u32), ("id", u32)]

        self._EventTypeSpec = EventTypeSpec
        self._EventHotKeyID = EventHotKeyID
        # OSStatus (*EventHandlerProcPtr)(EventHandlerCallRef, EventRef, void *userData)
        self._HandlerProc = ctypes.CFUNCTYPE(ctypes.c_int32, vp, vp, vp)

        lib.GetApplicationEventTarget.argtypes = []
        lib.GetApplicationEventTarget.restype = vp
        lib.InstallEventHandler.argtypes = [
            vp,
            self._HandlerProc,
            ctypes.c_size_t,  # ItemCount
            ctypes.POINTER(EventTypeSpec),
            vp,
            ctypes.POINTER(vp),
        ]
        lib.InstallEventHandler.restype = ctypes.c_int32
        lib.RemoveEventHandler.argtypes = [vp]
        lib.RemoveEventHandler.restype = ctypes.c_int32
        lib.RegisterEventHotKey.argtypes = [u32, u32, EventHotKeyID, vp, u32, ctypes.POINTER(vp)]
        lib.RegisterEventHotKey.restype = ctypes.c_int32
        lib.UnregisterEventHotKey.argtypes = [vp]
        lib.UnregisterEventHotKey.restype = ctypes.c_int32
        lib.GetEventParameter.argtypes = [
            vp,
            u32,
            u32,
            ctypes.POINTER(u32),
            ctypes.c_size_t,  # ByteCount
            ctypes.POINTER(ctypes.c_size_t),
            vp,
        ]
        lib.GetEventParameter.restype = ctypes.c_int32
        self._lib = lib
        # The C side only holds a raw function pointer: without this reference the
        # trampoline would be garbage-collected and the next hotkey would crash.
        self._proc: Any = None

    def install_handler(self, handler: Callable[[Any], int]) -> tuple[int, Any]:
        ctypes = self._ctypes

        def trampoline(_call_ref: Any, event: Any, _user_data: Any) -> int:
            try:
                return int(handler(event))
            except Exception:  # an exception must never unwind into Carbon
                log.exception("Hotkey event handler failed")
                return _EVENT_NOT_HANDLED_ERR

        proc = self._HandlerProc(trampoline)
        spec = self._EventTypeSpec(_EVENT_CLASS_KEYBOARD, _EVENT_HOTKEY_PRESSED)
        ref = ctypes.c_void_p()
        status = self._lib.InstallEventHandler(
            self._lib.GetApplicationEventTarget(),
            proc,
            1,
            ctypes.byref(spec),
            None,
            ctypes.byref(ref),
        )
        if status == _NO_ERR:
            self._proc = proc
        return int(status), ref

    def remove_handler(self, ref: Any) -> int:
        status = int(self._lib.RemoveEventHandler(ref))
        self._proc = None
        return status

    def register_hotkey(
        self, keycode: int, modifiers: int, signature: int, hotkey_id: int
    ) -> tuple[int, Any]:
        ctypes = self._ctypes
        ref = ctypes.c_void_p()
        status = self._lib.RegisterEventHotKey(
            keycode,
            modifiers,
            self._EventHotKeyID(signature, hotkey_id),
            self._lib.GetApplicationEventTarget(),
            0,
            ctypes.byref(ref),
        )
        return int(status), ref

    def unregister_hotkey(self, ref: Any) -> int:
        return int(self._lib.UnregisterEventHotKey(ref))

    def hotkey_id(self, event: Any) -> tuple[int, int, int]:
        """``(status, signature, id)`` of the hot key that produced ``event``."""
        ctypes = self._ctypes
        hk = self._EventHotKeyID()
        status = self._lib.GetEventParameter(
            event,
            _EVENT_PARAM_DIRECT_OBJECT,
            _TYPE_EVENT_HOTKEY_ID,
            None,
            ctypes.sizeof(hk),
            None,
            ctypes.byref(hk),
        )
        return int(status), int(hk.signature), int(hk.id)


class MacHotkeyManager(_NativeHotkeyManager):
    """Carbon ``RegisterEventHotKey`` hotkeys.

    Unlike an event tap this needs neither Accessibility nor Input Monitoring
    permission. Carbon is not thread-safe, so every call must come from the main
    thread (calls from other threads are refused with a warning); callbacks run on
    the main thread, dispatched by the Cocoa run loop Qt is already running.

    Key codes identify physical ANSI key positions, so on non-US layouts
    punctuation hotkeys refer to the key at the US position.
    """

    name: ClassVar[str] = "carbon"

    def __init__(self, carbon: Any = None) -> None:
        super().__init__()
        self._carbon = carbon  # injectable for tests; created lazily otherwise
        self._handler_ref: Any = None
        self._refs: dict[int, Any] = {}

    @staticmethod
    def _on_main_thread(what: str) -> bool:
        if threading.current_thread() is threading.main_thread():
            return True
        log.warning("macOS hotkeys must be %s from the main thread", what)
        return False

    def _backend_start(self) -> bool:
        if self._handler_ref is not None:
            return True
        if not self._on_main_thread("registered"):
            return False
        if self._carbon is None:
            try:
                self._carbon = _Carbon()
            except (OSError, AttributeError) as exc:
                log.warning("Carbon hot-key API unavailable: %s", exc)
                self.note = "The Carbon hot-key API could not be loaded."
                return False
        status, ref = self._carbon.install_handler(self._on_event)
        if status != _NO_ERR:
            log.warning("InstallEventHandler failed (OSStatus %s)", status)
            return False
        self._handler_ref = ref
        return True

    def _backend_stop(self) -> None:
        if self._handler_ref is None or not self._on_main_thread("stopped"):
            return
        for ref in self._refs.values():  # normally empty: stop() unregisters first
            self._carbon.unregister_hotkey(ref)
        self._refs.clear()
        self._carbon.remove_handler(self._handler_ref)
        self._handler_ref = None

    def _native_register(self, binding: _Binding) -> bool:
        if not self._on_main_thread("registered"):
            return False
        keycode = _MAC_KEYCODES.get(binding.hotkey.key)
        label = format_hotkey(binding.hotkey, macos=True)
        if keycode is None:
            log.warning("Key %s does not exist on macOS keyboards", label)
            return False
        status, ref = self._carbon.register_hotkey(
            keycode, _mac_modifiers(binding.hotkey.modifiers), _HOTKEY_SIGNATURE, binding.id
        )
        if status != _NO_ERR:
            if status == _EVENT_HOTKEY_EXISTS_ERR:
                log.warning("Hotkey %s is already registered", label)
            else:
                log.warning("RegisterEventHotKey(%s) failed (OSStatus %s)", label, status)
            return False
        self._refs[binding.id] = ref
        return True

    def _native_unregister(self, binding: _Binding) -> None:
        ref = self._refs.pop(binding.id, None)
        if ref is not None and self._on_main_thread("unregistered"):
            status = self._carbon.unregister_hotkey(ref)
            if status != _NO_ERR:
                log.debug("UnregisterEventHotKey failed (OSStatus %s)", status)

    def _on_event(self, event: Any) -> int:
        status, signature, hotkey_id = self._carbon.hotkey_id(event)
        if status != _NO_ERR or signature != _HOTKEY_SIGNATURE:
            return _EVENT_NOT_HANDLED_ERR  # someone else's hot key: let it propagate
        self._dispatch(hotkey_id)
        return _NO_ERR


# ----------------------------------------------------------------------------- X11
# Values from X.h, duplicated so the pure logic below is testable without python-xlib.
_X_SHIFT_MASK = 1 << 0
_X_LOCK_MASK = 1 << 1
_X_CONTROL_MASK = 1 << 2
_X_MOD1_MASK = 1 << 3  # Alt
_X_MOD2_MASK = 1 << 4  # NumLock on virtually every keymap
_X_MOD4_MASK = 1 << 6  # Super
_X_KEY_PRESS = 2
_X_KEY_RELEASE = 3
_X_MAPPING_NOTIFY = 34
_X_MAPPING_MODIFIER = 0
_X_MAPPING_KEYBOARD = 1
_X_GRAB_MODE_ASYNC = 1

_X_MODIFIERS = {
    "ctrl": _X_CONTROL_MASK,
    "alt": _X_MOD1_MASK,
    "shift": _X_SHIFT_MASK,
    "meta": _X_MOD4_MASK,
}
_X_RELEVANT_MASK = _X_SHIFT_MASK | _X_CONTROL_MASK | _X_MOD1_MASK | _X_MOD4_MASK

_X_KEYSYM_NAMES: dict[str, str] = {
    **{ch: ch for ch in _LETTERS + _DIGITS},
    **{f"f{n}": f"F{n}" for n in range(1, 25)},
    "space": "space",
    "enter": "Return",
    "tab": "Tab",
    "escape": "Escape",
    "backspace": "BackSpace",
    "insert": "Insert",
    "delete": "Delete",
    "home": "Home",
    "end": "End",
    "pageup": "Prior",
    "pagedown": "Next",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    "minus": "minus",
    "equal": "equal",
    "bracketleft": "bracketleft",
    "bracketright": "bracketright",
    "backslash": "backslash",
    "semicolon": "semicolon",
    "quote": "apostrophe",
    "grave": "grave",
    "comma": "comma",
    "period": "period",
    "slash": "slash",
}


def _x11_modifiers(modifiers: Iterable[str]) -> int:
    mask = 0
    for m in modifiers:
        mask |= _X_MODIFIERS[m]
    return mask


def _x11_lock_variants(numlock_mask: int) -> tuple[int, ...]:
    """Modifier masks to grab in addition to the real ones.

    A passive grab matches the modifier state exactly, so without these variants a
    hotkey would silently stop working whenever Caps Lock or Num Lock is on.
    """
    variants: list[int] = []
    for mask in (0, _X_LOCK_MASK, numlock_mask, _X_LOCK_MASK | numlock_mask):
        if mask not in variants:
            variants.append(mask)
    return tuple(variants)


def _x11_string_to_keysym(name: str) -> int:
    from Xlib import XK

    return int(XK.string_to_keysym(name))


def _x11_open_display(display_name: str | None) -> Any:
    from Xlib import display as xdisplay

    return xdisplay.Display(display_name)


class _XErrorCatcher:
    """``onerror`` handler for python-xlib requests (same protocol as ``Xlib.error.CatchError``).

    Returning a true value tells python-xlib the error was handled, so a failed grab
    (``BadAccess``: another client owns the combination) is recorded here instead of
    being reported asynchronously on stderr.
    """

    def __init__(self) -> None:
        self.error: Any = None

    def __call__(self, error: Any, request: Any) -> int:
        if self.error is None:
            self.error = error
        return 1


class X11HotkeyManager(_ThreadedHotkeyManager):
    """Passive key grabs on the X11 root window, served by a python-xlib thread.

    The thread owns its own ``Display`` connection. Its loop waits in ``select`` on
    that connection and a wake-up pipe (0.2 s timeout as a safety net), so it is
    idle between key presses and :meth:`stop` returns promptly.
    """

    name: ClassVar[str] = "x11"
    _SELECT_TIMEOUT_S: ClassVar[float] = 0.2

    def __init__(
        self,
        display_name: str | None = None,
        *,
        note: str | None = None,
        display_factory: Callable[[str | None], Any] | None = None,
    ) -> None:
        super().__init__(note)
        self._display_name = display_name
        self._display_factory = display_factory or _x11_open_display
        self._display: Any = None
        self._root: Any = None
        self._numlock_mask = _X_MOD2_MASK
        self._pipe_lock = threading.Lock()
        self._wake_r = -1
        self._wake_w = -1
        # Only touched on the hotkey thread:
        self._grabs: dict[int, tuple[int, int]] = {}  # binding id → (keycode, modifier mask)
        self._lookup: dict[tuple[int, int], int] = {}  # (keycode, modifier mask) → binding id
        self._held: set[int] = set()
        self._last_release: dict[int, int] = {}

    # ------------------------------------------------------------- lifecycle
    def _before_thread_start(self) -> bool:
        with self._pipe_lock:
            if self._wake_r < 0:
                try:
                    read_fd, write_fd = os.pipe()
                except OSError as exc:
                    log.warning("Cannot create the hotkey wake-up pipe: %s", exc)
                    return False
                try:
                    os.set_blocking(read_fd, False)
                    os.set_blocking(write_fd, False)
                except (OSError, AttributeError) as exc:  # not selectable here (not POSIX)
                    os.close(read_fd)
                    os.close(write_fd)
                    log.warning("Cannot configure the hotkey wake-up pipe: %s", exc)
                    return False
                self._wake_r, self._wake_w = read_fd, write_fd
        return True

    def _thread_setup(self) -> bool:
        try:
            display = self._display_factory(self._display_name)
        except Exception as exc:
            log.warning("Cannot connect to the X server for hotkeys: %s", exc)
            self.note = f"Could not connect to the X server: {exc}"
            return False
        display.set_error_handler(self._on_x_error)
        self._display = display
        self._root = display.screen().root
        self._numlock_mask = self._find_numlock_mask()
        return True

    def _thread_loop(self) -> None:
        import select

        display = self._display
        while not self._stopping.is_set():
            self._run_pending_calls()
            # python-xlib may already have buffered events (e.g. read during a sync),
            # which select() would not report, so drain before waiting.
            while display.pending_events():
                self._handle_event(display.next_event())
            try:
                readable, _, _ = select.select(
                    [display.fileno(), self._wake_r], [], [], self._SELECT_TIMEOUT_S
                )
            except (OSError, ValueError) as exc:
                log.warning("Hotkey thread stopping: %s", exc)
                break
            if self._wake_r in readable:
                self._drain_wake_pipe()

    def _thread_teardown(self) -> None:
        display = self._display
        if display is not None:
            try:
                for binding_id in list(self._grabs):
                    self._ungrab(binding_id)
                display.sync()
            except Exception:
                log.debug("Ungrabbing hotkeys failed", exc_info=True)
            try:
                display.close()
            except Exception:
                log.debug("Closing the X display failed", exc_info=True)
        self._display = None
        self._root = None
        self._grabs.clear()
        self._lookup.clear()
        self._held.clear()
        self._last_release.clear()
        with self._pipe_lock:
            for fd in (self._wake_r, self._wake_w):
                if fd >= 0:
                    with contextlib.suppress(OSError):
                        os.close(fd)
            self._wake_r = self._wake_w = -1

    def _wake(self) -> bool:
        with self._pipe_lock:
            if self._wake_w < 0:
                return False
            try:
                os.write(self._wake_w, b"\0")
            except BlockingIOError:
                return True  # pipe full: the thread is due to wake up anyway
            except OSError:
                return False
            return True

    def _drain_wake_pipe(self) -> None:
        # Non-blocking: read until empty (BlockingIOError is an OSError).
        with contextlib.suppress(OSError):
            while os.read(self._wake_r, 4096):
                pass

    # --------------------------------------------------------- registrations
    def _native_register(self, binding: _Binding) -> bool:
        return bool(self._call_on_thread(lambda: self._grab_binding(binding), False))

    def _native_unregister(self, binding: _Binding) -> None:
        self._call_on_thread(lambda: self._ungrab_binding(binding.id), None)

    def _keycode_for(self, key: str) -> int:
        keysym = _x11_string_to_keysym(_X_KEYSYM_NAMES[key])
        if not keysym:
            return 0
        return int(self._display.keysym_to_keycode(keysym))

    def _find_numlock_mask(self) -> int:
        try:
            keycode = int(self._display.keysym_to_keycode(_x11_string_to_keysym("Num_Lock")))
            if keycode:
                for index, keycodes in enumerate(self._display.get_modifier_mapping()):
                    if keycode in list(keycodes):
                        return 1 << index
        except Exception:
            log.debug("Could not determine the NumLock modifier", exc_info=True)
        return _X_MOD2_MASK

    def _grab_binding(self, binding: _Binding) -> bool:
        label = format_hotkey(binding.hotkey, macos=False)
        keycode = self._keycode_for(binding.hotkey.key)
        if not keycode:
            log.warning("Key for hotkey %s is not on the current keyboard layout", label)
            return False
        mods = _x11_modifiers(binding.hotkey.modifiers)
        if not self._grab(keycode, mods):
            log.warning("Hotkey %s is already grabbed by another application", label)
            return False
        self._grabs[binding.id] = (keycode, mods)
        self._lookup[(keycode, mods)] = binding.id
        return True

    def _ungrab_binding(self, binding_id: int) -> None:
        self._ungrab(binding_id)
        self._display.flush()

    def _grab(self, keycode: int, mods: int) -> bool:
        catcher = _XErrorCatcher()
        for variant in _x11_lock_variants(self._numlock_mask):
            self._root.grab_key(
                keycode,
                mods | variant,
                True,
                _X_GRAB_MODE_ASYNC,
                _X_GRAB_MODE_ASYNC,
                onerror=catcher,
            )
        # Grab errors arrive asynchronously; a round trip guarantees they were seen.
        self._display.sync()
        if catcher.error is None:
            return True
        # Release whichever variants did succeed so nothing half-registered lingers.
        quiet = _XErrorCatcher()
        for variant in _x11_lock_variants(self._numlock_mask):
            self._root.ungrab_key(keycode, mods | variant, onerror=quiet)
        self._display.sync()
        return False

    def _ungrab(self, binding_id: int) -> None:
        grab = self._grabs.pop(binding_id, None)
        self._held.discard(binding_id)
        self._last_release.pop(binding_id, None)
        if grab is None:
            return
        self._lookup.pop(grab, None)
        keycode, mods = grab
        quiet = _XErrorCatcher()
        for variant in _x11_lock_variants(self._numlock_mask):
            self._root.ungrab_key(keycode, mods | variant, onerror=quiet)

    def _regrab_all(self) -> None:
        """Re-establish grabs after a keyboard layout or modifier mapping change."""
        with self._table_lock:
            bindings = [b for b in self._by_id.values() if b.id in self._grabs]
        for binding in bindings:
            self._ungrab(binding.id)
        self._numlock_mask = self._find_numlock_mask()
        for binding in bindings:
            if not self._grab_binding(binding):
                log.warning(
                    "Hotkey %s stopped working after a keyboard layout change",
                    format_hotkey(binding.hotkey, macos=False),
                )

    # ------------------------------------------------------------------ events
    def _handle_event(self, event: Any) -> None:
        kind = event.type
        if kind == _X_MAPPING_NOTIFY:
            if event.request in (_X_MAPPING_KEYBOARD, _X_MAPPING_MODIFIER):
                self._display.refresh_keyboard_mapping(event)
                self._regrab_all()
            return
        if kind not in (_X_KEY_PRESS, _X_KEY_RELEASE):
            return
        binding_id = self._lookup.get((int(event.detail), int(event.state) & _X_RELEVANT_MASK))
        if binding_id is None:
            return
        if kind == _X_KEY_RELEASE:
            self._held.discard(binding_id)
            self._last_release[binding_id] = int(event.time)
            return
        # X auto-repeat sends Release+Press pairs with identical timestamps while a
        # key is held; fire once per physical press (the MOD_NOREPEAT equivalent).
        last_release = self._last_release.get(binding_id)
        if binding_id in self._held:
            return
        if last_release is not None and (int(event.time) - last_release) & 0xFFFFFFFF <= 1:
            return
        self._held.add(binding_id)
        self._dispatch(binding_id)

    @staticmethod
    def _on_x_error(error: Any, request: Any = None) -> None:
        log.debug("X error in hotkey connection: %s", error)


# --------------------------------------------------------------------------- factory
def _is_wayland_session() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or bool(
        os.environ.get("WAYLAND_DISPLAY")
    )


_WAYLAND_NOTE = (
    "Wayland does not allow applications to register global hotkeys. Bind "
    "'eye-tracker ctl toggle', 'eye-tracker ctl privacy-toggle' or "
    "'eye-tracker ctl calibrate' to shortcuts in your desktop's keyboard settings."
)
_XWAYLAND_NOTE = (
    "Running under Wayland: hotkeys are grabbed through XWayland and may only fire "
    "while an X11 window has focus. For reliable shortcuts bind 'eye-tracker ctl …' "
    "commands in your desktop's keyboard settings."
)


def create_hotkey_manager() -> HotkeyManager:
    """Return the best hotkey manager for this system (never raises)."""
    try:
        if sys.platform == "win32":
            return WindowsHotkeyManager()
        if sys.platform == "darwin":
            return MacHotkeyManager()
        if not os.environ.get("DISPLAY"):
            if _is_wayland_session():
                return HotkeyManager(note=_WAYLAND_NOTE)
            return HotkeyManager(note="No X11 display is available for global hotkeys.")
        try:
            import Xlib  # noqa: F401  (availability probe only)
        except ImportError:
            return HotkeyManager(note="python-xlib is not installed; global hotkeys are off.")
        note = _XWAYLAND_NOTE if _is_wayland_session() else None
        return X11HotkeyManager(note=note)
    except Exception as exc:  # defensive: hotkeys are optional
        log.warning("Global hotkeys unavailable: %s", exc)
        return HotkeyManager(note=f"Global hotkeys unavailable: {exc}")


__all__ = [
    "KEYS",
    "MODIFIERS",
    "Hotkey",
    "HotkeyCallback",
    "HotkeyManager",
    "MacHotkeyManager",
    "WindowsHotkeyManager",
    "X11HotkeyManager",
    "create_hotkey_manager",
    "format_hotkey",
    "parse_hotkey",
]
