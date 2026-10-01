"""A minimal UI Automation client through hand-written ctypes COM calls.

Only what the pane providers need is bound. For Windows Terminal: find the
descendants of a window with a given class name, read four properties of each
(runtime id, bounding rectangle, keyboard focus, off-screen) and give one of
them the keyboard focus. For desktop apps built on Chromium:
:class:`ControlView`, a step-by-step walk of the control view of one window
(first/last child, next/previous sibling), the focused element, and a few
properties per element (control type, class name, automation id, keyboard
focusability, process id). Names, values and text of elements are never read.
There is no pywin32 or comtypes: interface pointers are plain integers and
methods are called through their vtable slot.

Vtable slots
------------
The indices below come from ``UIAutomationClient.h`` (Windows SDK
10.0.26100), counted from the C ``...Vtbl`` structs, where slots 0-2 are
IUnknown's ``QueryInterface``, ``AddRef`` and ``Release``. COM interfaces are
immutable, so they hold for every Windows version that has the interface.
``IUIAutomation2`` (Windows 8 and later) extends ``IUIAutomation``: its own
methods follow slot 57, the last one of ``IUIAutomation``.

Timeouts
--------
``CUIAutomation8`` hands out ``IUIAutomation2``, whose connection and
transaction timeouts are set to :data:`CONNECTION_TIMEOUT_MS` and
:data:`TRANSACTION_TIMEOUT_MS` (UIA's defaults are 2 s and 20 s), so a
terminal that hangs makes a call fail quickly instead of blocking the pane
worker. Without ``CUIAutomation8`` the plain ``CUIAutomation`` is used with
UIA's defaults; the pane worker's own call timeout then still counts slow
calls as failures.

Threads
-------
COM is initialised as multithreaded (MTA) on each calling thread, once per
thread, the first time a :class:`UiAutomation` is used there; it is never
uninitialised (the pane worker's thread lives for the session). A thread
already initialised as single-threaded (a Qt GUI thread) cannot be used:
:class:`ComError` is raised. One automation object is created per
:class:`UiAutomation` and shared by its MTA threads; :meth:`UiAutomation.close`
releases it.

Every interface pointer obtained is released, also when a call fails; a failed
HRESULT raises :class:`ComError` (an :class:`OSError`).

The module imports on every platform: nothing Windows-specific is loaded until
a method is called, and then off Windows :class:`OSError` is raised.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import sys
import threading
from collections.abc import Callable
from ctypes import POINTER, byref, c_int, c_long, c_uint, c_ulong, c_ushort, c_void_p
from dataclasses import dataclass
from typing import Any

from ..types import Rect

log = logging.getLogger(__name__)

CLSID_CUIAUTOMATION = "{ff48dba4-60ef-4201-aa87-54103eef594e}"
IID_IUIAUTOMATION = "{30cbe57d-d9d0-452a-ab13-7ac5ac4825ee}"
CLSID_CUIAUTOMATION8 = "{e22ad333-b25f-460c-83d0-0581107395c9}"
IID_IUIAUTOMATION2 = "{34723aff-0c9d-49d0-9896-7ab52df8cd8a}"

#: Connection and transaction timeouts set on ``IUIAutomation2`` (milliseconds).
CONNECTION_TIMEOUT_MS = 200
TRANSACTION_TIMEOUT_MS = 500

CLSCTX_INPROC_SERVER = 0x1
COINIT_MULTITHREADED = 0x0
RPC_E_CHANGED_MODE = 0x80010106
TREE_SCOPE_DESCENDANTS = 0x4
UIA_CLASS_NAME_PROPERTY_ID = 30012
#: Control type ids (``UIA_...ControlTypeId`` in ``UIAutomationClient.h``).
UIA_EDIT_CONTROL_TYPE_ID = 50004
UIA_GROUP_CONTROL_TYPE_ID = 50026
UIA_DOCUMENT_CONTROL_TYPE_ID = 50030
UIA_PANE_CONTROL_TYPE_ID = 50033
VT_I4 = 3
VT_BSTR = 8
GA_ROOT = 2

# IUnknown
_RELEASE = 2
# IUIAutomation (IUIAutomation2 adds 58-63 after it)
_UIA_ELEMENT_FROM_HANDLE = 6
_UIA_GET_FOCUSED_ELEMENT = 8
_UIA_GET_CONTROL_VIEW_WALKER = 14
_UIA_CREATE_PROPERTY_CONDITION = 23
_UIA2_GET_CONNECTION_TIMEOUT = 60
_UIA2_PUT_CONNECTION_TIMEOUT = 61
_UIA2_GET_TRANSACTION_TIMEOUT = 62
_UIA2_PUT_TRANSACTION_TIMEOUT = 63
# IUIAutomationTreeWalker
_WALKER_FIRST_CHILD = 4
_WALKER_LAST_CHILD = 5
_WALKER_NEXT_SIBLING = 6
_WALKER_PREVIOUS_SIBLING = 7
# IUIAutomationElement
_EL_SET_FOCUS = 3
_EL_GET_RUNTIME_ID = 4
_EL_FIND_ALL = 6
_EL_PROCESS_ID = 20
_EL_CONTROL_TYPE = 21
_EL_HAS_KEYBOARD_FOCUS = 26
_EL_IS_KEYBOARD_FOCUSABLE = 27
_EL_AUTOMATION_ID = 29
_EL_CLASS_NAME = 30
_EL_IS_OFFSCREEN = 38
_EL_BOUNDING_RECTANGLE = 43
# IUIAutomationElementArray
_ARRAY_LENGTH = 3
_ARRAY_GET_ELEMENT = 4

#: Longest runtime id accepted (real ones have 2-4 parts).
_MAX_RUNTIME_ID = 32
#: Longest class name or automation id read (characters); the rest is cut off.
_MAX_BSTR = 4096
E_POINTER = 0x80004003


class ComError(OSError):
    """A COM call returned a failed HRESULT (``hresult`` is unsigned)."""

    def __init__(self, what: str, hresult: int) -> None:
        self.hresult = hresult & 0xFFFFFFFF
        super().__init__(f"{what} failed (HRESULT 0x{self.hresult:08X})")


@dataclass(frozen=True, slots=True)
class UiaElement:
    """What is read of one UI Automation element (no names, no text)."""

    #: ``GetRuntimeId``: unique among the elements on the desktop while the
    #: element exists; ``None`` if it could not be read.
    runtime_id: tuple[int, ...] | None
    #: Bounding rectangle in physical pixels, global coordinates; ``None`` if empty.
    rect: Rect | None
    has_keyboard_focus: bool
    offscreen: bool


class GUID(ctypes.Structure):
    _fields_ = [
        ("data1", c_ulong),
        ("data2", c_ushort),
        ("data3", c_ushort),
        ("data4", ctypes.c_ubyte * 8),
    ]


class RECT(ctypes.Structure):
    _fields_ = [("left", c_long), ("top", c_long), ("right", c_long), ("bottom", c_long)]


class VARIANT(ctypes.Structure):
    """``VARIANT`` with its union as two pointers (16 bytes on x86, 24 on x64)."""

    _fields_ = [
        ("vt", c_ushort),
        ("reserved1", c_ushort),
        ("reserved2", c_ushort),
        ("reserved3", c_ushort),
        ("value", c_void_p),
        ("extra", c_void_p),
    ]


def _check(hr: int, what: str) -> None:
    if hr < 0:
        raise ComError(what, hr)


# ------------------------------------------------------------- DLL binding
def _load_dll(name: str) -> Any:
    if sys.platform != "win32":
        raise OSError("UI Automation is only available on Windows")
    return ctypes.WinDLL(name)


class _Api:
    """The few exported functions used, with explicit prototypes."""

    def __init__(self) -> None:
        # Private WinDLL instances: prototypes declared here cannot clash with
        # other code using ctypes.windll.
        ole32 = _load_dll("ole32")
        oleaut32 = _load_dll("oleaut32")
        user32 = _load_dll("user32")
        hresult = c_long  # not ctypes.HRESULT: that raises instead of returning

        def bind(dll: Any, name: str, restype: Any, *argtypes: Any) -> Any:
            fn = getattr(dll, name)
            fn.restype = restype
            fn.argtypes = list(argtypes)
            return fn

        self.CoInitializeEx = bind(ole32, "CoInitializeEx", hresult, c_void_p, c_ulong)
        self.CoCreateInstance = bind(
            ole32,
            "CoCreateInstance",
            hresult,
            POINTER(GUID),
            c_void_p,
            c_ulong,
            POINTER(GUID),
            POINTER(c_void_p),
        )
        self.CLSIDFromString = bind(
            ole32, "CLSIDFromString", hresult, ctypes.c_wchar_p, POINTER(GUID)
        )
        self.SysAllocString = bind(oleaut32, "SysAllocString", c_void_p, ctypes.c_wchar_p)
        self.SysFreeString = bind(oleaut32, "SysFreeString", None, c_void_p)
        self.SysStringLen = bind(oleaut32, "SysStringLen", c_uint, c_void_p)
        self.SafeArrayGetDim = bind(oleaut32, "SafeArrayGetDim", c_uint, c_void_p)
        self.SafeArrayGetElemsize = bind(oleaut32, "SafeArrayGetElemsize", c_uint, c_void_p)
        self.SafeArrayGetVartype = bind(
            oleaut32, "SafeArrayGetVartype", hresult, c_void_p, POINTER(c_ushort)
        )
        self.SafeArrayGetLBound = bind(
            oleaut32, "SafeArrayGetLBound", hresult, c_void_p, c_uint, POINTER(c_long)
        )
        self.SafeArrayGetUBound = bind(
            oleaut32, "SafeArrayGetUBound", hresult, c_void_p, c_uint, POINTER(c_long)
        )
        self.SafeArrayAccessData = bind(
            oleaut32, "SafeArrayAccessData", hresult, c_void_p, POINTER(c_void_p)
        )
        self.SafeArrayUnaccessData = bind(oleaut32, "SafeArrayUnaccessData", hresult, c_void_p)
        self.SafeArrayDestroy = bind(oleaut32, "SafeArrayDestroy", hresult, c_void_p)
        self.GetForegroundWindow = bind(user32, "GetForegroundWindow", c_void_p)
        self.GetAncestor = bind(user32, "GetAncestor", c_void_p, c_void_p, c_uint)

    def guid(self, text: str) -> GUID:
        value = GUID()
        _check(self.CLSIDFromString(text, byref(value)), "CLSIDFromString")
        return value


def _api() -> _Api:
    """The bound DLL functions (loaded once; :class:`OSError` off Windows)."""
    if sys.platform != "win32":
        raise OSError("UI Automation is only available on Windows")
    return _loaded_api()


@functools.cache
def _loaded_api() -> _Api:
    return _Api()


def _method(ptr: int, index: int, *argtypes: Any, restype: Any = c_long) -> Callable[..., Any]:
    """Slot ``index`` of the vtable of interface pointer ``ptr``, bound to ``ptr``."""
    if sys.platform != "win32":
        raise OSError("COM is only available on Windows")
    vtable = ctypes.cast(c_void_p(ptr), POINTER(POINTER(c_void_p))).contents
    # ctypes caches the prototype classes, so this is cheap after the first call.
    function = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(vtable[index])
    return lambda *args: function(ptr, *args)


def _release(ptr: int | None) -> None:
    if ptr:
        _method(ptr, _RELEASE, restype=c_ulong)()


def _out_ptr(ptr: int, index: int, what: str, *args: Any, argtypes: tuple[Any, ...] = ()) -> int:
    """Call a method whose last parameter is an out interface pointer; 0 if null."""
    out = c_void_p()
    hr = _method(ptr, index, *argtypes, POINTER(c_void_p))(*args, byref(out))
    _check(hr, what)
    return out.value or 0


def _read_bool(ptr: int, index: int, what: str) -> bool:
    return bool(_read_int(ptr, index, what))


def _read_int(ptr: int, index: int, what: str) -> int:
    value = c_int()
    _check(_method(ptr, index, POINTER(c_int))(byref(value)), what)
    return int(value.value)


def _read_bstr(api: _Api, ptr: int, index: int, what: str) -> str:
    """Call a ``get_...`` method returning a ``BSTR``, which is freed; ``""`` if null."""
    out = c_void_p()
    _check(_method(ptr, index, POINTER(c_void_p))(byref(out)), what)
    bstr = out.value
    if not bstr:
        return ""
    try:
        length = min(int(api.SysStringLen(bstr)), _MAX_BSTR)
        return ctypes.wstring_at(bstr, length)
    finally:
        api.SysFreeString(bstr)


def _read_rect(element: int) -> Rect | None:
    rect = RECT()
    _check(
        _method(element, _EL_BOUNDING_RECTANGLE, POINTER(RECT))(byref(rect)),
        "get_CurrentBoundingRectangle",
    )
    width, height = rect.right - rect.left, rect.bottom - rect.top
    return Rect(rect.left, rect.top, width, height) if width > 0 and height > 0 else None


def _read_runtime_id(api: _Api, element: int) -> tuple[int, ...] | None:
    psa = c_void_p()
    hr = _method(element, _EL_GET_RUNTIME_ID, POINTER(c_void_p))(byref(psa))
    if hr < 0:
        return None
    return read_int_safearray(api, psa.value or 0) or None


def read_int_safearray(api: _Api, psa: int) -> tuple[int, ...] | None:
    """The values of a one-dimensional ``SAFEARRAY`` of 32-bit integers, which is
    destroyed. ``None`` if it is anything else."""
    if not psa:
        return None
    try:
        if api.SafeArrayGetDim(psa) != 1:
            return None
        vt = c_ushort()
        if api.SafeArrayGetVartype(psa, byref(vt)) >= 0:
            if vt.value != VT_I4:
                return None
        elif api.SafeArrayGetElemsize(psa) != ctypes.sizeof(c_int):
            return None
        lower, upper = c_long(), c_long()
        _check(api.SafeArrayGetLBound(psa, 1, byref(lower)), "SafeArrayGetLBound")
        _check(api.SafeArrayGetUBound(psa, 1, byref(upper)), "SafeArrayGetUBound")
        count = upper.value - lower.value + 1
        if count <= 0 or count > _MAX_RUNTIME_ID:
            return None
        data = c_void_p()
        _check(api.SafeArrayAccessData(psa, byref(data)), "SafeArrayAccessData")
        try:
            if not data.value:
                return None
            return tuple(int(v) for v in (c_int * count).from_address(data.value))
        finally:
            api.SafeArrayUnaccessData(psa)
    finally:
        api.SafeArrayDestroy(psa)


# ------------------------------------------------------------------ client
class UiAutomation:
    """A UI Automation client (see the module documentation for threads).

    Creating one loads nothing: the automation object is created on first use,
    on the calling thread.
    """

    def __init__(
        self,
        *,
        connection_timeout_ms: int = CONNECTION_TIMEOUT_MS,
        transaction_timeout_ms: int = TRANSACTION_TIMEOUT_MS,
    ) -> None:
        self._connection_timeout_ms = connection_timeout_ms
        self._transaction_timeout_ms = transaction_timeout_ms
        self._lock = threading.Lock()
        self._threads = threading.local()
        self._uia = 0
        self._has_timeouts = False
        self._conditions: dict[str, int] = {}

    # ---------------------------------------------------------------- public
    def find(self, hwnd: int, class_name: str) -> list[UiaElement]:
        """The descendants of window ``hwnd`` whose class name is ``class_name``."""
        found: list[UiaElement] = []

        def collect(_element: int, info: UiaElement) -> bool:
            found.append(info)
            return False

        self._each(hwnd, class_name, collect)
        return found

    def focus(self, hwnd: int, class_name: str, match: Callable[[UiaElement], bool]) -> bool:
        """Give the keyboard focus to the first such descendant ``match`` accepts.

        False when none does; :class:`ComError` when ``SetFocus`` fails.
        """
        done: list[bool] = []

        def set_focus(element: int, info: UiaElement) -> bool:
            if not match(info):
                return False
            _check(_method(element, _EL_SET_FOCUS)(), "IUIAutomationElement::SetFocus")
            done.append(True)
            return True

        self._each(hwnd, class_name, set_focus)
        return bool(done)

    def control_view(self, hwnd: int) -> ControlView:
        """A walk of the control view of window ``hwnd``; close it when done
        (it is a context manager)."""
        api = _api()
        return ControlView(api, self._automation(), hwnd)

    def foreground_window(self) -> int | None:
        """The top-level window in the foreground (``None`` if there is none)."""
        api = _api()
        hwnd = api.GetForegroundWindow()
        if not hwnd:
            return None
        root = api.GetAncestor(hwnd, GA_ROOT)
        return int(root or hwnd)

    def timeouts(self) -> tuple[int, int] | None:
        """(connection, transaction) timeouts in ms; ``None`` without IUIAutomation2."""
        uia = self._automation()
        if not self._has_timeouts:
            return None
        values = []
        for index, what in (
            (_UIA2_GET_CONNECTION_TIMEOUT, "get_ConnectionTimeout"),
            (_UIA2_GET_TRANSACTION_TIMEOUT, "get_TransactionTimeout"),
        ):
            value = c_ulong()
            _check(_method(uia, index, POINTER(c_ulong))(byref(value)), what)
            values.append(int(value.value))
        return values[0], values[1]

    def close(self) -> None:
        """Release the automation object and its conditions (call on an MTA thread)."""
        with self._lock:
            uia, self._uia = self._uia, 0
            conditions, self._conditions = self._conditions, {}
        for condition in conditions.values():
            _release(condition)
        _release(uia)

    # ------------------------------------------------------------- internals
    def _init_thread(self, api: _Api) -> None:
        if getattr(self._threads, "ready", False):
            return
        hr = api.CoInitializeEx(None, COINIT_MULTITHREADED)
        if hr & 0xFFFFFFFF == RPC_E_CHANGED_MODE:
            raise ComError("CoInitializeEx (thread is single-threaded)", hr)
        _check(hr, "CoInitializeEx")  # S_OK, or S_FALSE when already initialised
        self._threads.ready = True

    def _automation(self) -> int:
        api = _api()
        self._init_thread(api)
        with self._lock:
            if not self._uia:
                self._uia = self._create(api)
            return self._uia

    def _create(self, api: _Api) -> int:
        out = c_void_p()
        hr = api.CoCreateInstance(
            byref(api.guid(CLSID_CUIAUTOMATION8)),
            None,
            CLSCTX_INPROC_SERVER,
            byref(api.guid(IID_IUIAUTOMATION2)),
            byref(out),
        )
        if hr >= 0 and out.value:
            uia = int(out.value)
            self._has_timeouts = self._set_timeouts(uia)
            return uia
        log.debug("CUIAutomation8 not available (0x%08X); no UIA timeouts", hr & 0xFFFFFFFF)
        hr = api.CoCreateInstance(
            byref(api.guid(CLSID_CUIAUTOMATION)),
            None,
            CLSCTX_INPROC_SERVER,
            byref(api.guid(IID_IUIAUTOMATION)),
            byref(out),
        )
        _check(hr, "CoCreateInstance(CUIAutomation)")
        if not out.value:
            raise ComError("CoCreateInstance(CUIAutomation)", E_POINTER)
        self._has_timeouts = False
        return int(out.value)

    def _set_timeouts(self, uia: int) -> bool:
        try:
            put_connection = _method(uia, _UIA2_PUT_CONNECTION_TIMEOUT, c_ulong)
            _check(put_connection(self._connection_timeout_ms), "put_ConnectionTimeout")
            put_transaction = _method(uia, _UIA2_PUT_TRANSACTION_TIMEOUT, c_ulong)
            _check(put_transaction(self._transaction_timeout_ms), "put_TransactionTimeout")
        except ComError:
            log.debug("Could not set the UI Automation timeouts", exc_info=True)
            return False
        return True

    def _condition(self, api: _Api, uia: int, class_name: str) -> int:
        """A cached ``ClassName == class_name`` property condition."""
        with self._lock:
            cached = self._conditions.get(class_name)
        if cached:
            return cached
        bstr = api.SysAllocString(class_name)
        if not bstr:
            raise ComError("SysAllocString", 0x8007000E)  # E_OUTOFMEMORY
        try:
            value = VARIANT(vt=VT_BSTR, value=bstr)
            condition = _out_ptr(
                uia,
                _UIA_CREATE_PROPERTY_CONDITION,
                "IUIAutomation::CreatePropertyCondition",
                UIA_CLASS_NAME_PROPERTY_ID,
                value,
                argtypes=(c_int, VARIANT),
            )
        finally:
            api.SysFreeString(bstr)  # the condition keeps its own copy
        if not condition:
            raise ComError("IUIAutomation::CreatePropertyCondition", E_POINTER)
        with self._lock:
            if class_name in self._conditions:  # another thread was quicker
                _release(condition)
                return self._conditions[class_name]
            self._conditions[class_name] = condition
        return condition

    def _each(self, hwnd: int, class_name: str, visit: Callable[[int, UiaElement], bool]) -> None:
        """Call ``visit(element, info)`` for each matching descendant until it
        returns True. Every pointer is released before returning or raising."""
        api = _api()
        uia = self._automation()
        condition = self._condition(api, uia, class_name)
        root = _out_ptr(
            uia,
            _UIA_ELEMENT_FROM_HANDLE,
            "IUIAutomation::ElementFromHandle",
            c_void_p(hwnd),
            argtypes=(c_void_p,),
        )
        if not root:
            return
        try:
            array = _out_ptr(
                root,
                _EL_FIND_ALL,
                "IUIAutomationElement::FindAll",
                TREE_SCOPE_DESCENDANTS,
                c_void_p(condition),
                argtypes=(c_int, c_void_p),
            )
            if not array:
                return
            try:
                length = c_int()
                _check(
                    _method(array, _ARRAY_LENGTH, POINTER(c_int))(byref(length)),
                    "IUIAutomationElementArray::get_Length",
                )
                for i in range(max(0, length.value)):
                    element = _out_ptr(
                        array,
                        _ARRAY_GET_ELEMENT,
                        "IUIAutomationElementArray::GetElement",
                        i,
                        argtypes=(c_int,),
                    )
                    if not element:
                        continue
                    try:
                        if visit(element, self._describe(api, element)):
                            return
                    finally:
                        _release(element)
            finally:
                _release(array)
        finally:
            _release(root)

    @staticmethod
    def _describe(api: _Api, element: int) -> UiaElement:
        return UiaElement(
            runtime_id=_read_runtime_id(api, element),
            rect=_read_rect(element),
            has_keyboard_focus=_read_bool(
                element, _EL_HAS_KEYBOARD_FOCUS, "get_CurrentHasKeyboardFocus"
            ),
            offscreen=_read_bool(element, _EL_IS_OFFSCREEN, "get_CurrentIsOffscreen"),
        )


# -------------------------------------------------------------- tree walk
@dataclass(frozen=True, slots=True)
class UiaNode:
    """An element handed out by a :class:`ControlView` (valid until it closes)."""

    ptr: int


class ControlView:
    """Step through the control view of one window, one element at a time.

    Every element handed out is held until :meth:`close` (or the end of a
    ``with`` block) releases it, together with the tree walker; nothing is
    released earlier, so a caller bounds what it holds by bounding its walk.
    Only the properties below are read: never names, values or text.
    """

    def __init__(self, api: _Api, uia: int, hwnd: int) -> None:
        self._api = api
        self._uia = uia
        self._held: list[int] = []
        self._walker = 0
        self._closed = False
        self.root: UiaNode | None = None
        try:
            self._walker = _out_ptr(
                uia, _UIA_GET_CONTROL_VIEW_WALKER, "IUIAutomation::get_ControlViewWalker"
            )
            if not self._walker:
                raise ComError("IUIAutomation::get_ControlViewWalker", E_POINTER)
            self.root = self._hold(
                _out_ptr(
                    uia,
                    _UIA_ELEMENT_FROM_HANDLE,
                    "IUIAutomation::ElementFromHandle",
                    c_void_p(hwnd),
                    argtypes=(c_void_p,),
                )
            )
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> ControlView:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def held(self) -> int:
        """How many elements are held (released by :meth:`close`)."""
        return len(self._held)

    def close(self) -> None:
        """Release every element handed out and the walker (idempotent)."""
        self._closed = True
        held, self._held = self._held, []
        walker, self._walker = self._walker, 0
        for ptr in reversed(held):
            _release(ptr)
        _release(walker)

    # ------------------------------------------------------------ navigation
    def first_child(self, node: UiaNode) -> UiaNode | None:
        return self._step(node, _WALKER_FIRST_CHILD, "GetFirstChildElement")

    def last_child(self, node: UiaNode) -> UiaNode | None:
        return self._step(node, _WALKER_LAST_CHILD, "GetLastChildElement")

    def next_sibling(self, node: UiaNode) -> UiaNode | None:
        return self._step(node, _WALKER_NEXT_SIBLING, "GetNextSiblingElement")

    def previous_sibling(self, node: UiaNode) -> UiaNode | None:
        return self._step(node, _WALKER_PREVIOUS_SIBLING, "GetPreviousSiblingElement")

    def focused(self) -> UiaNode | None:
        """The element with the keyboard focus, on the whole desktop."""
        self._ensure_open()
        return self._hold(
            _out_ptr(self._uia, _UIA_GET_FOCUSED_ELEMENT, "IUIAutomation::GetFocusedElement")
        )

    # ------------------------------------------------------------ properties
    def control_type(self, node: UiaNode) -> int:
        return _read_int(self._ptr(node), _EL_CONTROL_TYPE, "get_CurrentControlType")

    def process_id(self, node: UiaNode) -> int:
        return _read_int(self._ptr(node), _EL_PROCESS_ID, "get_CurrentProcessId")

    def class_name(self, node: UiaNode) -> str:
        return _read_bstr(self._api, self._ptr(node), _EL_CLASS_NAME, "get_CurrentClassName")

    def automation_id(self, node: UiaNode) -> str:
        return _read_bstr(self._api, self._ptr(node), _EL_AUTOMATION_ID, "get_CurrentAutomationId")

    def is_keyboard_focusable(self, node: UiaNode) -> bool:
        return _read_bool(
            self._ptr(node), _EL_IS_KEYBOARD_FOCUSABLE, "get_CurrentIsKeyboardFocusable"
        )

    def is_offscreen(self, node: UiaNode) -> bool:
        return _read_bool(self._ptr(node), _EL_IS_OFFSCREEN, "get_CurrentIsOffscreen")

    def rect(self, node: UiaNode) -> Rect | None:
        """Bounding rectangle in physical pixels, global coordinates; ``None`` if empty."""
        return _read_rect(self._ptr(node))

    def runtime_id(self, node: UiaNode) -> tuple[int, ...] | None:
        return _read_runtime_id(self._api, self._ptr(node))

    def set_focus(self, node: UiaNode) -> None:
        """Give ``node`` the keyboard focus (:class:`ComError` if that fails)."""
        _check(_method(self._ptr(node), _EL_SET_FOCUS)(), "IUIAutomationElement::SetFocus")

    # ------------------------------------------------------------- internals
    def _ensure_open(self) -> None:
        if self._closed:
            raise ComError("ControlView (closed)", E_POINTER)

    def _ptr(self, node: UiaNode) -> int:
        self._ensure_open()
        return node.ptr

    def _hold(self, ptr: int) -> UiaNode | None:
        if not ptr:
            return None
        self._held.append(ptr)
        return UiaNode(ptr)

    def _step(self, node: UiaNode, index: int, what: str) -> UiaNode | None:
        ptr = self._ptr(node)
        return self._hold(
            _out_ptr(
                self._walker,
                index,
                f"IUIAutomationTreeWalker::{what}",
                c_void_p(ptr),
                argtypes=(c_void_p,),
            )
        )
