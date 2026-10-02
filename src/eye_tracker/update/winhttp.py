"""The update check's connection on Windows: WinHTTP, through ctypes.

This is the only module of Eye Tracker that touches the network, and the only
exception in ``scripts/check_privacy.py`` (``SOURCE_ALLOWLIST``). It is used for
one thing, the opt-in update check (see ``docs/privacy.md``), and it is as small
as it can be:

* WinHTTP is part of Windows. Using it adds no library to the app (no OpenSSL,
  no Python ``ssl`` or ``urllib``), so the bundle scan stays as strict as before.
* Certificates are checked by Windows against its own root store, and proxy
  settings are the system's (automatic detection). Nothing here turns a check
  off.
* Redirects are switched off in WinHTTP: :class:`~eye_tracker.update.fetch.Fetcher`
  follows them one hop at a time and checks every host.
* Only ``GET`` is sent, with a ``User-Agent`` that names the app and its
  version, and an ``Accept`` header. No cookie, no identifier, no body.

Not used on other platforms: :func:`~eye_tracker.update.fetch.supported` is false
there and this module is never imported.
"""

from __future__ import annotations

import ctypes
import functools
import sys
from ctypes import wintypes

from .. import REPO_URL, __version__
from .fetch import NetworkError

USER_AGENT = f"EyeTracker/{__version__} (+{REPO_URL})"

_ACCESS_TYPE_AUTOMATIC_PROXY = 4
_FLAG_SECURE = 0x00800000
_OPTION_REDIRECT_POLICY = 88
_REDIRECT_POLICY_NEVER = 0
_QUERY_CONTENT_LENGTH = 5
_QUERY_LOCATION = 33
_QUERY_STATUS_CODE = 19
_QUERY_FLAG_NUMBER = 0x20000000
_PORT = 443
#: Milliseconds: resolve, connect, send, receive.
_TIMEOUTS = (10_000, 10_000, 15_000, 30_000)


@functools.cache
def _winhttp() -> ctypes.WinDLL:
    """The WinHTTP library with typed signatures (loaded on first use)."""
    if sys.platform != "win32":  # pragma: no cover - guarded by fetch.supported()
        raise NetworkError("WinHTTP exists on Windows only.")
    api = ctypes.WinDLL("winhttp", use_last_error=True)
    handle, bool_, dword, wstr = wintypes.LPVOID, wintypes.BOOL, wintypes.DWORD, wintypes.LPCWSTR
    ptr = ctypes.c_void_p
    signatures = {
        "WinHttpOpen": (handle, [wstr, dword, wstr, wstr, dword]),
        "WinHttpSetTimeouts": (
            bool_,
            [handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
        ),
        "WinHttpConnect": (handle, [handle, wstr, wintypes.WORD, dword]),
        "WinHttpOpenRequest": (handle, [handle, wstr, wstr, wstr, wstr, ptr, dword]),
        "WinHttpSetOption": (bool_, [handle, dword, ptr, dword]),
        "WinHttpSendRequest": (bool_, [handle, wstr, dword, ptr, dword, dword, ctypes.c_size_t]),
        "WinHttpReceiveResponse": (bool_, [handle, ptr]),
        "WinHttpQueryHeaders": (bool_, [handle, dword, wstr, ptr, ctypes.POINTER(dword), ptr]),
        "WinHttpReadData": (bool_, [handle, ptr, dword, ctypes.POINTER(dword)]),
        "WinHttpCloseHandle": (bool_, [handle]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(api, name)
        function.restype = restype
        function.argtypes = argtypes
    return api


def _fail(what: str) -> NetworkError:
    code = ctypes.get_last_error()
    return NetworkError(f"{what} (Windows error {code}).")


class _WinHttpResponse:
    """An answered request; :meth:`read` hands out the body."""

    def __init__(self, request: int, connection: int) -> None:
        self._api = _winhttp()
        self._request: int | None = request
        self._connection: int | None = connection
        self.status = self._number(_QUERY_STATUS_CODE) or 0
        self.content_length: int | None = self._number(_QUERY_CONTENT_LENGTH)
        self.location: str | None = self._text(_QUERY_LOCATION)

    def _number(self, level: int) -> int | None:
        value = wintypes.DWORD(0)
        size = wintypes.DWORD(ctypes.sizeof(value))
        ok = self._api.WinHttpQueryHeaders(
            self._request,
            level | _QUERY_FLAG_NUMBER,
            None,
            ctypes.byref(value),
            ctypes.byref(size),
            None,
        )
        return int(value.value) if ok else None

    def _text(self, level: int) -> str | None:
        buffer = ctypes.create_unicode_buffer(2048)
        size = wintypes.DWORD(ctypes.sizeof(buffer))
        ok = self._api.WinHttpQueryHeaders(
            self._request, level, None, buffer, ctypes.byref(size), None
        )
        return buffer.value if ok and buffer.value else None

    def read(self, size: int) -> bytes:
        if self._request is None:
            return b""
        buffer = ctypes.create_string_buffer(size)
        read = wintypes.DWORD(0)
        if not self._api.WinHttpReadData(self._request, buffer, size, ctypes.byref(read)):
            raise _fail("The connection was interrupted")
        return buffer.raw[: read.value]

    def close(self) -> None:
        for name in ("_request", "_connection"):
            handle = getattr(self, name)
            if handle is not None:
                self._api.WinHttpCloseHandle(handle)
                setattr(self, name, None)


class WinHttpTransport:
    """One WinHTTP session; every :meth:`open` is a single GET without redirects."""

    def __init__(self) -> None:
        self._api = _winhttp()
        self._session: int | None = self._api.WinHttpOpen(
            USER_AGENT, _ACCESS_TYPE_AUTOMATIC_PROXY, None, None, 0
        )
        if not self._session:
            raise _fail("Could not start a connection")
        self._api.WinHttpSetTimeouts(self._session, *_TIMEOUTS)

    def open(self, host: str, path: str, accept: str | None) -> _WinHttpResponse:
        api = self._api
        connection = api.WinHttpConnect(self._session, host, _PORT, 0)
        if not connection:
            raise _fail(f"Could not reach {host}")
        request = api.WinHttpOpenRequest(connection, "GET", path, None, None, None, _FLAG_SECURE)
        if not request:
            error = _fail("Could not start a request")
            api.WinHttpCloseHandle(connection)
            raise error
        try:
            policy = wintypes.DWORD(_REDIRECT_POLICY_NEVER)
            api.WinHttpSetOption(
                request, _OPTION_REDIRECT_POLICY, ctypes.byref(policy), ctypes.sizeof(policy)
            )
            headers = f"Accept: {accept}\r\n" if accept else None
            sent = api.WinHttpSendRequest(
                request, headers, 0xFFFFFFFF if headers else 0, None, 0, 0, 0
            )
            if not sent or not api.WinHttpReceiveResponse(request, None):
                raise _fail(f"Could not reach {host}")
            return _WinHttpResponse(request, connection)
        except BaseException:
            api.WinHttpCloseHandle(request)
            api.WinHttpCloseHandle(connection)
            raise

    def close(self) -> None:
        if self._session is not None:
            self._api.WinHttpCloseHandle(self._session)
            self._session = None
