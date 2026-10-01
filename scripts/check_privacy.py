#!/usr/bin/env python3
"""Privacy gate for Eye Tracker: the source tree and the frozen bundle.

Eye Tracker promises that it makes no network connections and never writes
camera frames to disk. This script enforces that promise mechanically, in two
places.

Source tree (default)
=====================

It parses (never imports) every Python file under ``src/eye_tracker`` and
reports:

``network-import``
    An import of a networking module (``socket``, ``_socket``, ``ssl``,
    ``http``, ``urllib``, ``requests``, ``multiprocessing.connection``, a
    telemetry SDK, ...), including dynamic imports with a literal name
    (``importlib.import_module``).
``qt-network``
    A Qt network class other than the local IPC ones (``QLocalServer`` and
    ``QLocalSocket``): ``QAbstractSocket``, ``QTcpSocket``, ``QUdpSocket``,
    ``QNetworkAccessManager``..., however it is reached, a wildcard import from
    ``QtNetwork``, or any Qt web/remote module (``QtWebSockets``, ``QtWebEngine*``...).
``network-api``
    A standard-library call that opens a connection or a server without an
    obviously networking import: asyncio's ``open_connection``,
    ``start_server``, ``loop.create_connection``, ``sock_connect``...,
    ``logging.handlers`` network handlers (``HTTPHandler``, ``SocketHandler``...)
    and ``logging.config.listen``.
``frame-write``
    Any use of an image/video encoder that could put a frame on disk or into a
    byte buffer (``imwrite``, ``imencode``, ``VideoWriter``).
``native-network``
    A native networking library loaded through ctypes (``ws2_32``,
    ``wininet``, ``winhttp``, ``libcurl``...), or raw socket and resolver calls
    made through ctypes (``libc.socket``, ``libc.connect``, ``getaddrinfo``,
    ``WSAStartup``...), which is how a raw socket could be created and polled
    with ``select`` without the ``socket`` module.
``network-command``
    A network command-line tool started as a subprocess (``curl``, ``wget``,
    ``ssh``...), also behind a shell or wrapper (``sh -c "curl ..."``,
    ``powershell -c "iwr ..."``, ``sudo wget``).
``macos-network``
    Apple's networking APIs as pyobjc exposes them, which the app's macOS code
    could reach through the Foundation and Core Foundation modules it loads: the
    URL-loading, host-name and Bonjour classes (``NSURLSession`` and its tasks,
    ``NSURLConnection``, ``NSURLDownload``, ``NSURLRequest``, ``NSHost``,
    ``NSNetService``...) and Core Foundation's host, socket and HTTP stream
    functions (``CFStreamCreatePairWithSocketToHost``...), however they are
    reached (attribute, import, ``getattr``, ``objc.lookUpClass``,
    ``NSClassFromString``); a call of a ``...WithContentsOfURL_`` selector
    (``NSData.dataWithContentsOfURL_``, ``NSString.stringWithContentsOfURL_...``,
    ``initWithContentsOfURL_``, ``NSImage.initByReferencingURL_``), which reads a
    web address as readily as a file, unless its URL is a file URL made with
    ``NSURL.fileURLWithPath_`` (in place, or in a name that only ever holds one);
    and ``objc.loadBundle`` (or ``NSBundle``) loading ``CFNetwork``, ``Network`` or
    ``WebKit``. Importing pyobjc's ``CFNetwork``, ``Network`` or ``WebKit``
    modules is a ``network-import``.
``syntax-error``
    A file that cannot be parsed, and therefore cannot be verified.

Frozen bundle (``--bundle DIR``)
================================

The source check cannot see what third-party native code does; the MediaPipe
runtime, for example, contained a usage-logging client that uploaded to Google.
The release workflow therefore also scans the PyInstaller output. Every native
binary (PE, ELF and Mach-O, found by their magic bytes, not by file name) is
parsed with the small readers below (no ``pefile``, ``readelf`` or ``otool``
needed), and the bundle fails on:

``telemetry-endpoint``
    Any file containing a known telemetry/usage-logging marker
    (``play.googleapis.com``, Clearcut, Firebase logging, Sentry, Segment...),
    including the zlib-compressed Python code inside the executables (see
    below) and the members of ZIP files. Never allow-listed.
``forbidden-package``
    A bundled Python package that must not ship (``mediapipe``, ``requests``,
    ``urllib3``, telemetry SDKs, any third-party network client...), found by
    its ``.dist-info`` folder, its package directory, or its modules in the
    executables' PYZ archive (also when vendored inside another package).
``network-plugin``
    A Qt plugin that talks to the network: the TLS, network-information and
    network-access backends, the TUIO touch listener (UDP), or the VNC/WebGL
    platform plugins (TCP servers). The spec removes them all.
``network-library``
    A binary that links a networking library (``WS2_32``, ``WININET``,
    ``WINHTTP``, ``DNSAPI``... on Windows; ``libcurl``, ``libssl``,
    ``libresolv``, ``libkrb5``... elsewhere; the ``CFNetwork``, ``Network``,
    ``GSS``, ``Kerberos``, ``LDAP`` and ``WebKit`` frameworks on macOS) or Qt's
    own ``QtNetwork``, and is not on the allow-list.
``network-symbol``
    An ELF or Mach-O binary that imports host-name resolution or remote
    connection functions (``getaddrinfo``, ``gethostbyname``...; on Linux and
    macOS sockets live in libc, so the imported *symbols* are what matter), or
    Foundation's URL-loading classes (``_OBJC_CLASS_$_NSURLSession``,
    ``NSURLConnection``, ``NSURLDownload``: Foundation is linked by every macOS
    binary, so the classes it imports are what matter) and is not on the
    allow-list. Plain ``socket``/``connect`` are not flagged: Qt, D-Bus and X11
    use them for local (AF_UNIX) connections.
``unreadable-binary``
    A native binary whose headers cannot be parsed, so it cannot be verified.
``unreadable-archive``
    Python code the gate cannot read: a PyInstaller archive (or ZIP file) that
    does not parse, or PyInstaller itself is not installed to read it.

PyInstaller stores every pure-Python module (the app's and all third-party
ones) zlib-compressed in the PYZ archive embedded in each executable, where a
plain byte scan sees nothing. The gate therefore opens that archive with
PyInstaller's own reader (``PyInstaller.archive.readers``) and checks every
module's name and decompressed code; the scripts and bootstrap modules stored
next to the PYZ, and files embedded in a one-file build, are checked too.
Standard-library networking modules (``http.client``, ``socket``...) are not
reported there: the standard library imports them itself, and the source check
keeps the app's own code away from them.

Some bundled libraries legitimately link networking code without ever using
it for remote connections (Qt's QLocalSocket lives in QtNetwork, CPython's
``_socket`` module is needed by ``psutil``, OpenCV's FFmpeg can read URLs that
the app refuses to open; on macOS, OpenCV's wheel also ships the protocol
libraries of Homebrew's FFmpeg build). They are listed in
:data:`BUNDLE_ALLOWLIST`, each with the exact indicators it may have and the
reason; anything else fails. Allow-listed findings are printed as notices so
every release log shows them. A library that is bundled but not needed is
removed by the PyInstaller spec instead of being allowed here.

Usage::

    python scripts/check_privacy.py                    # scan src/eye_tracker
    python scripts/check_privacy.py PATH ...           # scan other files or directories
    python scripts/check_privacy.py --bundle DIST_DIR  # scan a frozen build

Exit status: 0 when clean, 1 when violations were found, 2 on a usage error.

The source check is deliberately conservative: it has no allow-list comments.
If it flags something legitimate, change the rule here so the exception is
reviewed. It cannot see through values computed at runtime (a module name
built from variables, ``QImage.save`` on a camera frame); code review covers
those.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import os
import re
import struct
import sys
import zipfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "src" / "eye_tracker"

#: Top-level modules that exist to talk to the network (or only wrap it).
NETWORK_MODULES: frozenset[str] = frozenset(
    {
        # Standard library and its C accelerators.
        "_multiprocessing",  # closesocket/recv/send on Windows socket handles
        "_overlapped",  # WSAConnect/ConnectEx for asyncio on Windows
        "_socket",
        "_ssl",
        "asynchat",
        "asyncore",
        "ftplib",
        "http",
        "imaplib",
        "nntplib",
        "poplib",
        "smtpd",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "telnetlib",
        "urllib",
        "wsgiref",
        "xmlrpc",
        # Third-party network clients and servers.
        "aiohttp",
        "asyncssh",
        "boto3",
        "botocore",
        "dns",
        "googleapiclient",
        "grpc",
        "h11",
        "h2",
        "httpcore",
        "httplib2",
        "httpx",
        "paho",
        "paramiko",
        "pycurl",
        "requests",
        "tornado",
        "twisted",
        "urllib3",
        "websocket",
        "websockets",
        "zeroconf",
        "zmq",
        # pyobjc's wrappers of Apple's networking frameworks.
        "CFNetwork",
        "Network",
        "WebKit",
        # Telemetry, analytics and crash-reporting SDKs.
        "amplitude",
        "bugsnag",
        "datadog",
        "ddtrace",
        "firebase_admin",
        "mediapipe",  # ships a Clearcut usage logger that uploads to Google
        "mixpanel",
        "newrelic",
        "opentelemetry",
        "posthog",
        "rollbar",
        "sentry_sdk",
    }
)

#: Dotted modules whose parent package is harmless but which exist to connect
#: processes over sockets (``multiprocessing.connection.Client(("host", 443))``).
NETWORK_SUBMODULES: frozenset[str] = frozenset(
    {
        "google.cloud",
        "multiprocessing.connection",
        "multiprocessing.dummy.connection",
        "multiprocessing.managers",
    }
)

QT_BINDINGS: tuple[str, ...] = ("PySide6", "PySide2", "PyQt6", "PyQt5")

#: The only QtNetwork classes the app may use: local (same-machine) IPC.
QT_NETWORK_ALLOWED: frozenset[str] = frozenset({"QLocalServer", "QLocalSocket"})

#: QtNetwork classes for remote communication. The names are unique to Qt, so
#: they are flagged wherever they appear, however the module was reached.
QT_NETWORK_CLASSES: frozenset[str] = frozenset(
    {
        "QAbstractSocket",
        "QAuthenticator",
        "QDnsLookup",
        "QDtls",
        "QDtlsClientVerifier",
        "QHostAddress",
        "QHostInfo",
        "QHstsPolicy",
        "QHttp2Configuration",
        "QHttpMultiPart",
        "QHttpPart",
        "QNetworkAccessManager",
        "QNetworkCookieJar",
        "QNetworkDatagram",
        "QNetworkDiskCache",
        "QNetworkInformation",
        "QNetworkProxy",
        "QNetworkProxyFactory",
        "QNetworkReply",
        "QNetworkRequest",
        "QNetworkRequestFactory",
        "QRestAccessManager",
        "QSctpServer",
        "QSctpSocket",
        "QSslServer",
        "QSslSocket",
        "QTcpServer",
        "QTcpSocket",
        "QUdpSocket",
    }
)

#: Qt modules whose whole purpose is web or remote communication.
QT_BANNED_MODULES: frozenset[str] = frozenset(
    {
        "QtCoap",
        "QtGrpc",
        "QtHttpServer",
        "QtMqtt",
        "QtNetworkAuth",
        "QtRemoteObjects",
        "QtWebChannel",
        "QtWebEngine",
        "QtWebEngineCore",
        "QtWebEngineQuick",
        "QtWebEngineWidgets",
        "QtWebSockets",
        "QtWebView",
    }
)

#: Apple's networking classes and functions as pyobjc exposes them (Foundation and
#: Core Foundation, which the app's macOS code loads): loading URLs, resolving host
#: names, Bonjour, sockets to hosts. The names are unique to Apple's frameworks, so
#: they are flagged wherever they appear, however the module was reached.
MACOS_NETWORK_NAMES: frozenset[str] = frozenset(
    {
        "NSHost",
        "NSMutableURLRequest",
        "NSNetService",
        "NSNetServiceBrowser",
        "NSURLConnection",
        "NSURLDownload",
        "NSURLRequest",
        "NSURLSession",
        "CFHostCreateWithName",
        "CFHostStartInfoResolution",
        "CFReadStreamCreateForHTTPRequest",
        "CFReadStreamCreateForStreamedHTTPRequest",
        "CFSocketConnectToAddress",
        "CFStreamCreatePairWithSocketToHost",
        "CFURLCreateDataAndPropertiesFromResource",
    }
)
#: Families of the classes above (``NSURLSessionConfiguration``, ``NSURLSessionDataTask``,
#: ``NSURLConnectionDelegate``...).
_MACOS_NETWORK_PREFIXES: tuple[str, ...] = ("NSURLSession", "NSURLConnection")
#: Selectors (pyobjc spelling) that read whatever a URL names, a web address as
#: readily as a file: ``NSData.dataWithContentsOfURL_``,
#: ``NSString.stringWithContentsOfURL_encoding_error_``, ``initWithContentsOfURL_``,
#: ``NSImage.initByReferencingURL_``... They are allowed with a file URL made in place.
_MACOS_URL_READ = re.compile(r"[a-z](?:WithContentsOfURL|ByReferencingURL)_")
#: ``NSURL`` constructors that can only make a file URL (``fileURLWithPath_``...).
_FILE_URL_PREFIX = "fileURLWith"
#: ``NSURL`` constructors from a string: a file URL when the literal says ``file:``.
_URL_FROM_STRING: frozenset[str] = frozenset(
    {"URLWithString_", "URLWithString_relativeToURL_", "initWithString_"}
)
#: How pyobjc code reaches a class or a framework by name.
_OBJC_CLASS_LOOKUPS: frozenset[str] = frozenset(
    {"lookUpClass", "NSClassFromString", "objc_getClass", "objc_lookUpClass"}
)
_OBJC_BUNDLE_LOADERS: frozenset[str] = frozenset(
    {"loadBundle", "bundleWithPath_", "bundleWithIdentifier_"}
)
#: Frameworks that ``objc.loadBundle`` must not load (by name, path or identifier).
MACOS_NETWORK_BUNDLES: frozenset[str] = frozenset({"CFNetwork", "Network", "WebKit"})

#: Standard-library functions and classes that open connections or servers.
#: asyncio's live on loops, streams and the package, so they are matched by name.
NETWORK_APIS: frozenset[str] = frozenset(
    {
        # asyncio streams and event loops.
        "connect_accepted_socket",
        "create_connection",
        "create_datagram_endpoint",
        "create_server",
        "create_unix_connection",
        "create_unix_server",
        "open_connection",
        "open_unix_connection",
        "sock_accept",
        "sock_connect",
        "sock_recvfrom",
        "sock_sendto",
        "start_server",
        "start_unix_server",
        # logging.handlers classes that send records over the network.
        "DatagramHandler",
        "HTTPHandler",
        "SMTPHandler",
        "SocketHandler",
        "SysLogHandler",
    }
)

#: Fully qualified names that open a network server.
NETWORK_DOTTED_NAMES: frozenset[str] = frozenset({"logging.config.listen"})

#: Encoders that turn a frame into a file or a byte buffer.
FRAME_WRITE_NAMES: frozenset[str] = frozenset(
    {"imwrite", "imwritemulti", "imencode", "VideoWriter"}
)

#: Windows DLLs that provide networking (import names, lower case, no ".dll").
WINDOWS_NETWORK_DLLS: frozenset[str] = frozenset(
    {
        "cryptnet",
        "dnsapi",
        "httpapi",
        "icmp",
        "iphlpapi",
        "mpr",
        "mswsock",
        "netapi32",
        "rasapi32",
        "urlmon",
        "webio",
        "websocket",
        "winhttp",
        "wininet",
        "wldap32",
        "ws2_32",
        "wsock32",
    }
)

#: POSIX shared libraries (soname prefixes) and macOS frameworks for networking.
POSIX_NETWORK_LIBRARIES: tuple[str, ...] = (
    "libcurl",
    "libgnutls",
    "libgssapi",
    "libkrb5",
    "libldap",
    "libnghttp2",
    "libresolv",
    "libsoup",
    "libssl",
    "libwebsockets",
    "libzmq",
)
#: Apple's networking frameworks, the counterparts of libgssapi, libkrb5 and libldap,
#: and WebKit (a web browser engine).
MACOS_NETWORK_FRAMEWORKS: frozenset[str] = frozenset(
    {"CFNetwork", "GSS", "Kerberos", "LDAP", "Network", "WebKit"}
)

#: Library names the source check refuses to load through ctypes.
NATIVE_NETWORK_LIBS: frozenset[str] = WINDOWS_NETWORK_DLLS | {
    "cfnetwork",
    "curl",
    "libcurl",
    "libresolv",
    "libsoup",
}

#: Native resolver/connection functions whose names only exist for networking;
#: flagged on any object (``ws2.WSAStartup``, ``libc.getaddrinfo``...).
NATIVE_NETWORK_FUNCTIONS: frozenset[str] = frozenset(
    {
        "DnsQuery_A",
        "DnsQuery_W",
        "GetAddrInfoW",
        "HttpOpenRequestW",
        "HttpSendRequestW",
        "IcmpSendEcho",
        "IcmpSendEcho2",
        "InternetConnectW",
        "InternetOpenUrlW",
        "InternetOpenW",
        "URLDownloadToFileW",
        "URLOpenStreamW",
        "getaddrinfo",
        "gethostbyaddr",
        "gethostbyname",
        "gethostbyname2",
        "gethostbyname_r",
        "getnameinfo",
    }
)
#: Prefixes of native networking function families (``WSASocketW``, ``WinHttpOpen``...).
_NATIVE_NETWORK_FUNCTION_PREFIXES: tuple[str, ...] = (
    "WSA",
    "WinHttp",
    "curl_easy_",
    "curl_global_",
    "curl_multi_",
)
#: libc socket calls: harmless names elsewhere (``signal.connect``), so they are
#: flagged only on ctypes library handles.
NATIVE_SOCKET_CALLS: frozenset[str] = frozenset(
    {
        "accept",
        "accept4",
        "bind",
        "connect",
        "listen",
        "recvfrom",
        "recvmsg",
        "sendmsg",
        "sendto",
        "socket",
    }
)

#: Command-line tools that transfer data over the network.
NETWORK_COMMANDS: frozenset[str] = frozenset(
    {
        "aria2c",
        "bitsadmin",
        "certutil",
        "curl",
        "dig",
        "fetch",
        "ftp",
        "http",
        "https",
        "invoke-restmethod",
        "invoke-webrequest",
        "irm",
        "iwr",
        "lynx",
        "nc",
        "ncat",
        "netcat",
        "nslookup",
        "ping",
        "rsync",
        "scp",
        "sftp",
        "socat",
        "ssh",
        "telnet",
        "tftp",
        "w3m",
        "wget",
        "xh",
    }
)

#: Commands that run another command given as an argument (``sh -c "curl ..."``).
_COMMAND_WRAPPERS: frozenset[str] = frozenset(
    {
        "bash",
        "busybox",
        "cmd",
        "dash",
        "doas",
        "env",
        "flatpak-spawn",
        "nice",
        "nohup",
        "pkexec",
        "powershell",
        "pwsh",
        "sh",
        "start",
        "sudo",
        "timeout",
        "xargs",
        "zsh",
    }
)

#: Call names whose first argument is a command line.
_COMMAND_CALLS: frozenset[str] = frozenset(
    {
        "Popen",
        "ShellExecuteExW",
        "ShellExecuteW",
        "WinExec",
        "call",
        "check_call",
        "check_output",
        "create_subprocess_exec",
        "create_subprocess_shell",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execute",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "getoutput",
        "getstatusoutput",
        "popen",
        "posix_spawn",
        "posix_spawnp",
        "run",
        "setProgram",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "start",
        "startDetached",
        "startfile",
        "system",
        "which",
    }
)

#: ctypes calls that load a library (``find_library`` only returns its path).
_LIBRARY_LOADERS: frozenset[str] = frozenset(
    {"CDLL", "OleDLL", "PyDLL", "WinDLL", "LoadLibrary", "find_library"}
)
_HANDLE_LOADERS: frozenset[str] = _LIBRARY_LOADERS - {"find_library"}
#: ``ctypes.cdll.<lib>`` and friends are library handles as well.
_CTYPES_LIBRARY_LOADERS: frozenset[str] = frozenset({"cdll", "windll", "oledll", "pydll"})

_LOCAL_ONLY = " (only QLocalServer/QLocalSocket are allowed)"
_PYTHON_SUFFIXES = (".py", ".pyw")
_SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "node_modules"})
_SHELL_WORD = re.compile(r"[^\s;&|()`'\"<>$]+")


@dataclass(frozen=True, order=True)
class Violation:
    """One privacy rule broken at a specific source location."""

    path: Path
    line: int
    col: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{_display_path(self.path)}:{self.line}:{self.col}: {self.rule}: {self.message}"


@dataclass
class ScanResult:
    """Outcome of scanning a set of files."""

    files: list[Path] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


# --------------------------------------------------------------------------- helpers
def _display_path(path: Path) -> str:
    """Path relative to the working directory when possible (clickable in CI logs)."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _dotted_name(node: ast.AST) -> str | None:
    """``a.b.c`` for a Name/Attribute chain, ``None`` for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _call_name(node: ast.Call) -> str | None:
    """Last component of the called name (``subprocess.run`` → ``run``)."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_qt_network_module(module: str) -> bool:
    """True for exactly ``<binding>.QtNetwork`` (the module, not a class in it)."""
    parts = module.split(".")
    return len(parts) == 2 and parts[0] in QT_BINDINGS and parts[1] == "QtNetwork"


def _is_network_module(module: str) -> bool:
    if module.split(".", 1)[0] in NETWORK_MODULES:
        return True
    return any(module == sub or module.startswith(sub + ".") for sub in NETWORK_SUBMODULES)


def _string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _command_word(text: str) -> str:
    """Normalise a command token: ``C:\\tools\\curl.exe`` → ``curl``."""
    word = text.strip().strip("\"'").replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if word.endswith(suffix):
            word = word[: -len(suffix)]
    return word


def _network_tool_in(command_line: str) -> str | None:
    """The first network tool anywhere in a shell command line, or ``None``.

    Every word is checked, not only the first, so ``sudo wget ...``,
    ``sh -c "curl ..."`` and ``a && curl ...`` are all caught.
    """
    for word in _SHELL_WORD.findall(command_line):
        if _command_word(word) in NETWORK_COMMANDS:
            return word
    return None


def _native_lib_stem(text: str) -> str:
    stem = text.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".dll", ".dylib"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.split(".so", 1)[0]


def _is_native_network_lib(text: str) -> bool:
    stem = _native_lib_stem(text)
    if stem in NATIVE_NETWORK_LIBS or stem.startswith(POSIX_NETWORK_LIBRARIES):
        return True
    # macOS frameworks: ".../Network.framework/Network", ".../CFNetwork.framework/...",
    # ".../WebKit.framework/WebKit"
    path = text.replace("\\", "/").lower()
    return "network.framework" in path or "/webkit.framework" in path


def _is_native_network_function(name: str) -> bool:
    return name in NATIVE_NETWORK_FUNCTIONS or name.startswith(_NATIVE_NETWORK_FUNCTION_PREFIXES)


def _is_macos_network_name(name: str) -> bool:
    return name in MACOS_NETWORK_NAMES or name.startswith(_MACOS_NETWORK_PREFIXES)


def _is_macos_network_bundle(text: str) -> bool:
    """``"CFNetwork"``, ``".../WebKit.framework"`` or ``"com.apple.Network"``."""
    lowered = text.lower()
    return any(
        lowered in (name.lower(), f"com.apple.{name.lower()}")
        or f"/{name.lower()}.framework" in lowered
        for name in MACOS_NETWORK_BUNDLES
    )


# --------------------------------------------------------------------------- scanner
class _ModuleScanner:
    """Collects violations for one parsed module."""

    def __init__(self, path: Path, tree: ast.Module) -> None:
        self.path = path
        self.tree = tree
        self.violations: list[Violation] = []
        # Names bound to the QtNetwork module in this file ("QtNetwork", "qn", ...).
        self.qt_network_aliases: set[str] = {f"{binding}.QtNetwork" for binding in QT_BINDINGS}
        # Names and attribute chains bound to ctypes library handles
        # ("libc", "self._ws2"), and functions that return one.
        self.library_handles: set[str] = set()
        self.handle_functions: set[str] = set()
        # Names and attribute chains that only ever hold a file URL ("url = NSURL.
        # fileURLWithPath_(path)"), and the attributes that are called directly.
        self.file_urls: set[str] = set()
        self.called: set[int] = set()

    def report(self, node: ast.AST, rule: str, message: str) -> None:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", -1) + 1
        self.violations.append(Violation(self.path, line, col, rule, message))

    def run(self) -> list[Violation]:
        # Imports can appear in any scope, so aliases are collected first and
        # attribute uses are checked in a second pass.
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                self._check_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._check_import_from(node)
        self._collect_library_handles()
        self._collect_file_urls()
        self.called = {id(n.func) for n in ast.walk(self.tree) if isinstance(n, ast.Call)}

        exempt: set[int] = set()
        for node in ast.walk(self.tree):
            if id(node) in exempt:
                continue
            if isinstance(node, ast.Attribute):
                exempt |= self._check_attribute(node)
            elif isinstance(node, ast.Name):
                self._check_name(node)
            elif isinstance(node, ast.Call):
                self._check_call(node)
            elif isinstance(node, ast.List | ast.Tuple):
                self._check_command_sequence(node)
        return sorted(set(self.violations))

    # ------------------------------------------------------------------ imports
    def _check_module_name(self, node: ast.AST, module: str, how: str) -> None:
        if _is_network_module(module):
            self.report(node, "network-import", f"{how} of networking module '{module}'")
            return
        parts = module.split(".")
        if parts[0] in QT_BINDINGS and len(parts) >= 2:
            if parts[1] in QT_BANNED_MODULES:
                self.report(node, "qt-network", f"{how} of Qt web/remote module '{module}'")
            elif parts[1] == "QtNetwork" and len(parts) > 2 and parts[2] not in QT_NETWORK_ALLOWED:
                self.report(node, "qt-network", f"{how} of '{module}'{_LOCAL_ONLY}")

    def _check_import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module_name(node, alias.name, "import")
            # "import PySide6.QtNetwork" binds "PySide6" (covered by the default
            # prefixes); "import PySide6.QtNetwork as qn" binds the module itself.
            if alias.asname and _is_qt_network_module(alias.name):
                self.qt_network_aliases.add(alias.asname)

    def _check_import_from(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module:
            return  # relative imports stay inside the package
        module = node.module
        self._check_module_name(node, module, "import")
        for alias in node.names:
            if alias.name in FRAME_WRITE_NAMES:
                self.report(node, "frame-write", f"import of '{alias.name}' from '{module}'")
            full = f"{module}.{alias.name}"
            # "from multiprocessing import connection" imports a network submodule.
            if not _is_network_module(module) and _is_network_module(full):
                self.report(node, "network-import", f"import of networking module '{full}'")
            if alias.name in NETWORK_APIS or full in NETWORK_DOTTED_NAMES:
                self.report(node, "network-api", f"import of '{alias.name}' from '{module}'")
            if _is_macos_network_name(alias.name):
                self.report(node, "macos-network", f"import of '{alias.name}' from '{module}'")
        parts = module.split(".")
        if parts[0] not in QT_BINDINGS:
            return
        if len(parts) == 1:
            for alias in node.names:
                if alias.name in QT_BANNED_MODULES:
                    self.report(node, "qt-network", f"import of Qt web module '{alias.name}'")
                elif alias.name == "QtNetwork":
                    self.qt_network_aliases.add(alias.asname or alias.name)
        elif parts[1] == "QtNetwork" and len(parts) == 2:
            for alias in node.names:
                if alias.name == "*":
                    self.report(node, "qt-network", f"wildcard import from '{module}'")
                elif alias.name not in QT_NETWORK_ALLOWED:
                    self.report(node, "qt-network", f"'{alias.name}' from '{module}'{_LOCAL_ONLY}")

    # ----------------------------------------------------------- ctypes handles
    def _is_handle_expr(self, node: ast.AST) -> bool:
        """``CDLL(...)``, ``ctypes.cdll.x``, a tracked handle, or a call returning one."""
        if isinstance(node, ast.Call):
            name = _call_name(node)
            return name in _HANDLE_LOADERS or name in self.handle_functions
        dotted = _dotted_name(node)
        if dotted is None:
            return False
        if dotted in self.library_handles:
            return True
        # ctypes.cdll.msvcrt, windll.ws2_32: an attribute of a library loader.
        return any(part in _CTYPES_LIBRARY_LOADERS for part in dotted.split(".")[:-1])

    def _collect_library_handles(self) -> None:
        """Find names bound to ctypes libraries, following simple re-bindings."""
        functions = [
            n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        assignments: list[tuple[list[ast.expr], ast.expr]] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                assignments.append((node.targets, node.value))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                assignments.append(([node.target], node.value))
        # A few rounds reach the fixpoint for any realistic chain of re-bindings.
        for _ in range(4):
            before = (len(self.library_handles), len(self.handle_functions))
            for function in functions:
                if any(
                    isinstance(sub, ast.Return)
                    and sub.value is not None
                    and self._is_handle_expr(sub.value)
                    for sub in ast.walk(function)
                ):
                    self.handle_functions.add(function.name)
            for targets, value in assignments:
                if self._is_handle_expr(value):
                    for target in targets:
                        dotted = _dotted_name(target)
                        if dotted is not None:
                            self.library_handles.add(dotted)
            if (len(self.library_handles), len(self.handle_functions)) == before:
                break

    # --------------------------------------------------------------- file URLs
    def _is_file_url(self, node: ast.AST | None) -> bool:
        """Whether ``node`` is certainly a file URL: ``NSURL.fileURLWithPath_(...)``,
        ``NSURL.URLWithString_("file:...")``, or a name that only ever holds one."""
        if isinstance(node, ast.Call):
            name = _call_name(node) or ""
            if name.startswith(_FILE_URL_PREFIX):
                return True
            literal = _string(node.args[0]) if node.args else None
            return name in _URL_FROM_STRING and (literal or "").lower().startswith("file:")
        dotted = _dotted_name(node) if node is not None else None
        return dotted is not None and dotted in self.file_urls

    def _collect_file_urls(self) -> None:
        """Names (and attribute chains) every assignment of which is a file URL."""
        values: dict[str, list[ast.expr]] = {}
        for node in ast.walk(self.tree):
            targets: list[ast.expr]
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr) and node.value:
                targets, value = [node.target], node.value
            else:
                continue
            for target in targets:
                dotted = _dotted_name(target)
                if dotted is not None:
                    values.setdefault(dotted, []).append(value)
        # A name assigned from another file-URL name: a few rounds reach the fixpoint.
        for _ in range(4):
            before = len(self.file_urls)
            self.file_urls |= {
                name
                for name, assigned in values.items()
                if all(self._is_file_url(value) for value in assigned)
            }
            if len(self.file_urls) == before:
                break

    def _check_url_read(self, node: ast.Call, name: str) -> None:
        """``NSData.dataWithContentsOfURL_(url)`` and its kind, unless ``url`` is a file URL."""
        if not self._is_file_url(node.args[0] if node.args else None):
            self.report(
                node,
                "macos-network",
                f"'{name}' reads whatever its URL names, a web address too "
                "(pass a file URL made with NSURL.fileURLWithPath_)",
            )

    # --------------------------------------------------------------- references
    def _check_attribute(self, node: ast.Attribute) -> set[int]:
        """Check one attribute access; returns ids of child nodes to skip."""
        if node.attr == "fourcc" and self._is_video_writer(node.value):
            # cv2.VideoWriter.fourcc("M", "J", "P", "G") only builds a codec code
            # (used to request MJPG from a camera); it writes nothing.
            return {id(node.value)}
        attr = node.attr
        if attr in FRAME_WRITE_NAMES:
            self.report(node, "frame-write", f"use of '{attr}' (frames must never be encoded)")
        if attr.lower() in NATIVE_NETWORK_LIBS:
            self.report(node, "native-network", f"native networking library '{attr}'")
        if _is_native_network_function(attr):
            self.report(node, "native-network", f"native networking function '{attr}'")
        elif attr in NATIVE_SOCKET_CALLS and self._is_handle_expr(node.value):
            self.report(node, "native-network", f"raw socket call '{attr}' through ctypes")
        base = _dotted_name(node.value)
        if base in self.qt_network_aliases and attr not in QT_NETWORK_ALLOWED:
            self.report(node, "qt-network", f"'{base}.{attr}'{_LOCAL_ONLY}")
        elif attr in QT_NETWORK_CLASSES:
            self.report(node, "qt-network", f"use of Qt network class '{attr}'{_LOCAL_ONLY}")
        if attr in NETWORK_APIS:
            self.report(node, "network-api", f"use of '{attr}' (opens a network connection)")
        if _is_macos_network_name(attr):
            self.report(node, "macos-network", f"use of '{attr}' (macOS networking)")
        elif _MACOS_URL_READ.search(attr) and id(node) not in self.called:
            # Called directly, the URL it is given is checked (see _check_call).
            self.report(node, "macos-network", f"'{attr}' reads whatever its URL names")
        dotted = _dotted_name(node)
        if dotted is not None:
            if dotted in NETWORK_DOTTED_NAMES:
                self.report(node, "network-api", f"use of '{dotted}' (opens a network server)")
            elif any(dotted == sub for sub in NETWORK_SUBMODULES):
                self.report(node, "network-import", f"use of networking module '{dotted}'")
        return set()

    @staticmethod
    def _is_video_writer(node: ast.AST) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr == "VideoWriter") or (
            isinstance(node, ast.Name) and node.id == "VideoWriter"
        )

    def _check_name(self, node: ast.Name) -> None:
        name = node.id
        if name in FRAME_WRITE_NAMES:
            self.report(node, "frame-write", f"use of '{name}' (frames must never be encoded)")
        if name in QT_NETWORK_CLASSES:
            self.report(node, "qt-network", f"use of Qt network class '{name}'{_LOCAL_ONLY}")
        if name in NETWORK_APIS:
            self.report(node, "network-api", f"use of '{name}' (opens a network connection)")
        if _is_macos_network_name(name):
            self.report(node, "macos-network", f"use of '{name}' (macOS networking)")

    # -------------------------------------------------------------------- calls
    def _check_call(self, node: ast.Call) -> None:
        name = _call_name(node)
        first = node.args[0] if node.args else None
        literal = _string(first)

        if name in {"import_module", "__import__"} and literal is not None:
            self._check_module_name(node, literal, "dynamic import")
            if _is_qt_network_module(literal):
                self.report(
                    node, "qt-network", f"dynamic import of '{literal}' (uses cannot be verified)"
                )

        if name == "getattr" and first is not None and len(node.args) >= 2:
            self._check_getattr(node, first, _string(node.args[1]))

        if name in _LIBRARY_LOADERS and literal is not None and _is_native_network_lib(literal):
            self.report(node, "native-network", f"loads native networking library {literal!r}")

        if name in _COMMAND_CALLS and literal is not None:
            tool = _network_tool_in(literal)
            if tool is not None:
                self.report(node, "network-command", f"runs network tool {tool!r}")

        if name is not None and _MACOS_URL_READ.search(name):
            self._check_url_read(node, name)
        if name in _OBJC_CLASS_LOOKUPS and literal is not None and _is_macos_network_name(literal):
            self.report(node, "macos-network", f"{name}({literal!r}) (macOS networking)")
        if name in _OBJC_BUNDLE_LOADERS:
            # objc.loadBundle("CFNetwork", globals(), bundle_path="/System/.../CFNetwork.framework")
            texts = [_string(arg) for arg in node.args] + [_string(k.value) for k in node.keywords]
            for text in texts:
                if text is not None and _is_macos_network_bundle(text):
                    self.report(node, "macos-network", f"{name} loads {text!r}")
                    break

    def _check_getattr(self, node: ast.Call, target_node: ast.AST, attr: str | None) -> None:
        target = _dotted_name(target_node)
        if target is not None and target in self.qt_network_aliases:
            if attr is None or attr not in QT_NETWORK_ALLOWED:
                shown = attr if attr is not None else "<dynamic>"
                self.report(node, "qt-network", f"getattr({target}, {shown!r})")
            return
        if attr is None:
            return
        if attr in QT_NETWORK_CLASSES:
            self.report(node, "qt-network", f"getattr(..., {attr!r}){_LOCAL_ONLY}")
        if attr in NETWORK_APIS:
            self.report(node, "network-api", f"getattr(..., {attr!r}) opens a network connection")
        if _is_macos_network_name(attr) or _MACOS_URL_READ.search(attr):
            self.report(node, "macos-network", f"getattr(..., {attr!r}) (macOS networking)")
        if _is_native_network_function(attr) or (
            attr in NATIVE_SOCKET_CALLS and self._is_handle_expr(target_node)
        ):
            self.report(node, "native-network", f"getattr(..., {attr!r}) through ctypes")

    def _check_command_sequence(self, node: ast.List | ast.Tuple) -> None:
        # ["curl", url] is flagged wherever it is built, so a command stored in a
        # variable before being passed to subprocess is caught too. Behind a
        # wrapper (["sh", "-c", "curl ..."], ["sudo", "wget", ...]) every
        # argument is a command line of its own.
        if not node.elts:
            return
        head = _string(node.elts[0])
        if head is None:
            return
        if _command_word(head) in NETWORK_COMMANDS:
            self.report(node, "network-command", f"command line runs network tool {head!r}")
            return
        if _command_word(head) not in _COMMAND_WRAPPERS:
            return
        for element in node.elts[1:]:
            text = _string(element)
            tool = _network_tool_in(text) if text is not None else None
            if tool is not None:
                self.report(
                    node, "network-command", f"command line runs network tool {tool!r} via {head!r}"
                )
                return


# ------------------------------------------------------------------ public API (source)
def scan_source(source: str | bytes, path: Path) -> list[Violation]:
    """Scan Python source code; ``path`` is only used for reporting."""
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError) as exc:
        line = getattr(exc, "lineno", None) or 0
        message = f"cannot be parsed, so it was not verified: {exc}"
        return [Violation(path, line, 1, "syntax-error", message)]
    return _ModuleScanner(path, tree).run()


def scan_file(path: Path) -> list[Violation]:
    """Scan one file (the source encoding cookie is honoured)."""
    return scan_source(path.read_bytes(), path)


def iter_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Python files under the given files/directories, in a stable order."""
    for root in paths:
        if root.is_file():
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for filename in sorted(filenames):
                if filename.endswith(_PYTHON_SUFFIXES):
                    yield Path(dirpath) / filename


def scan_paths(paths: Iterable[Path]) -> ScanResult:
    """Scan every Python file under ``paths``."""
    result = ScanResult()
    for path in iter_python_files(paths):
        result.files.append(path)
        result.violations.extend(scan_file(path))
    return result


# ====================================================================== frozen bundles
#: Byte strings that identify usage logging / telemetry clients. Matched
#: case-insensitively, as ASCII and as UTF-16LE, in every file of the bundle.
TELEMETRY_MARKERS: tuple[str, ...] = (
    "play.googleapis.com",  # Google Clearcut log endpoint (MediaPipe's usage logger)
    "clearcut",
    "firebaselogging",
    "crashlyticsreports",
    "app-measurement.com",
    "google-analytics.com",
    "googletagmanager.com",
    "ingest.sentry.io",
    "api.segment.io",
    "api.mixpanel.com",
    "api2.amplitude.com",
    "i.posthog.com",
    "app.posthog.com",
    "sessions.bugsnag.com",
    "api.rollbar.com",
    "dc.services.visualstudio.com",
)

#: Python distributions/packages that must never be part of a bundle.
FORBIDDEN_DISTRIBUTIONS: frozenset[str] = frozenset(
    {
        "aiohttp",
        "httpx",
        "mediapipe",
        "posthog",
        "requests",
        "sentry-sdk",
        "urllib3",
        "websockets",
    }
)

#: Networking modules that do not come with Python: finding one of their
#: modules in a bundle means a third-party network client (or telemetry SDK)
#: ships, possibly vendored inside another package.
THIRD_PARTY_NETWORK_MODULES: frozenset[str] = NETWORK_MODULES - sys.stdlib_module_names

#: Normalised names (see :func:`_library_key`) of Qt's network library on
#: Windows (``Qt6Network.dll``), Linux (``libQt6Network.so.6``) and macOS
#: (``QtNetwork.framework/.../QtNetwork``). Linking it is reported like any
#: other networking library, so a Qt module or plugin that gains networking
#: fails the gate until it is reviewed.
QT_NETWORK_LIBRARY_KEYS: frozenset[str] = frozenset(
    {"qt5network", "qt6network", "libqt5network", "libqt6network", "qtnetwork"}
)

#: Qt plugins that open network connections or listening sockets, as
#: ``(plugin group, file-name glob)`` pairs and why. The spec removes them.
QT_NETWORK_PLUGINS: tuple[tuple[str, str, str], ...] = (
    ("tls", "*", "a TLS backend (the app makes no remote connections)"),
    ("networkinformation", "*", "a network-reachability backend"),
    ("networkaccess", "*", "a QNetworkAccessManager backend"),
    ("generic", "*tuiotouch*", "the TUIO touch plugin listens on a UDP port"),
    ("platforms", "*vnc*", "the VNC platform is an unauthenticated TCP server"),
    ("platforms", "*webgl*", "the WebGL platform is an HTTP/WebSocket server"),
)

#: Imported functions (ELF / Mach-O) that resolve host names or open remote
#: connections. Plain ``socket``/``connect`` are omitted on purpose: they are
#: also how local (AF_UNIX) IPC works. On macOS, Foundation does HTTP(S) without
#: CFNetwork being linked: a binary that uses NSURLSession (or the older
#: NSURLConnection and NSURLDownload) imports its class symbol, stored here, like
#: every symbol, without the leading underscore.
NETWORK_SYMBOLS: frozenset[str] = frozenset(
    {
        "CFHostStartInfoResolution",
        "CFReadStreamCreateForHTTPRequest",
        "CFReadStreamCreateForStreamedHTTPRequest",
        "CFSocketConnectToAddress",
        "CFStreamCreatePairWithSocketToHost",
        "OBJC_CLASS_$_NSURLConnection",
        "OBJC_CLASS_$_NSURLDownload",
        "OBJC_CLASS_$_NSURLSession",
        "OBJC_CLASS_$_NSURLSessionConfiguration",
        "DNSServiceGetAddrInfo",
        "DNSServiceQueryRecord",
        "gethostbyaddr",
        "gethostbyaddr_r",
        "gethostbyname",
        "gethostbyname2",
        "gethostbyname2_r",
        "gethostbyname_r",
        "getaddrinfo",
        "getaddrinfo_a",
        "getnameinfo",
        "nw_connection_create",
        "res_nquery",
        "res_nsearch",
        "res_query",
        "res_search",
        "__res_query",
        "__res_search",
    }
)


@dataclass(frozen=True)
class AllowRule:
    """Networking indicators a bundled binary may have, and why.

    ``pattern`` is matched case-insensitively against the binary's path relative
    to the bundle, from the right like :meth:`pathlib.PurePath.match`
    (``"qt6network.dll"`` matches ``_internal/PySide6/Qt6Network.dll``).
    ``indicators`` are glob patterns for the normalised indicator keys that
    :func:`network_indicators` produces (``"ws2_32"``, ``"libgssapi_krb5"``,
    ``"getaddrinfo"``...); nothing else is allowed for that binary.
    """

    pattern: str
    indicators: frozenset[str]
    reason: str

    def matches(self, relative: str) -> bool:
        return PurePosixPath(relative.lower()).match(self.pattern.lower())

    def allows(self, key: str) -> bool:
        return any(fnmatch.fnmatchcase(key, indicator) for indicator in self.indicators)


def _allow(pattern: str, indicators: Iterable[str], reason: str) -> AllowRule:
    return AllowRule(pattern, frozenset(i.lower() for i in indicators), reason)


_RESOLVER = ("getaddrinfo", "getnameinfo", "gethostby*")
_WHY_PYTHON = (
    "CPython's socket support: psutil and parts of the standard library import the "
    "socket module; the app's own code never does (enforced by the source check)"
)
_WHY_QT_NETWORK = (
    "QtNetwork provides QLocalServer/QLocalSocket for the single-instance and "
    "'eye-tracker ctl' IPC; its TCP/TLS backends and plugins are removed by the spec"
)
_WHY_FFMPEG = (
    "OpenCV's FFmpeg could read network URLs, but the app opens only local files (it "
    "refuses URLs and protocols for every video source), and FFmpeg's whitelist lets a "
    "local file refer only to file, crypto and data URLs"
)
# OpenCV's macOS wheel ships FFmpeg as Homebrew builds it, with every protocol and
# filter library, in cv2/.dylibs (cv2/__dot__dylibs inside the .app). Measured on
# the macos-14 runner (otool -L, nm -m): cv2's extension module links libavformat
# and libavdevice, and through them all of these, so dyld needs every one of them
# to import cv2. The spec leaves out the libraries that nothing binds a symbol to.
_WHY_FFMPEG_PROTOCOL = (
    "network protocol library of the FFmpeg in OpenCV's macOS wheel (rist://, srt://, "
    "sftp:// and zmq:// URLs): the app opens only local files, and FFmpeg's whitelist "
    "lets a local file refer only to file, crypto and data URLs, so it is never used"
)
_WHY_FFMPEG_LINKS = (
    "Homebrew's FFmpeg links each of its libraries against the libraries of all its "
    "components; libavdevice binds no symbol to these, libavfilter binds only its zmq "
    "and azmq filters (commands over ZeroMQ), and OpenCV builds no filter graph"
)
_WHY_TESSERACT = (
    "Tesseract OCR, linked by FFmpeg's libavfilter for its ocr filter, which OpenCV "
    "never runs; Tesseract uses libcurl only to fetch images given as URLs and the "
    "resolver only for its ScrollView debugging viewer"
)
_WHY_XCB_MACOS = (
    "XCB client library for FFmpeg's X11 screen-grabbing device (libavdevice); OpenCV "
    "captures the camera with AVFoundation and reads files with FFmpeg's demuxers, so "
    "no X display is ever contacted"
)
_WHY_PYOBJC = (
    "pyobjc-core converts struct sockaddr values to and from Python tuples for Cocoa "
    "methods that take them (getaddrinfo resolves a host name only when Python code "
    "passes one); the app's macOS code calls no such method"
)
_WHY_PYOBJC_NETSERVICE = (
    "pyobjc's NSNetService.addresses() wrapper formats Bonjour addresses as numbers "
    "(getnameinfo with NI_NUMERICHOST, no lookup); the app uses no NSNetService"
)
_WHY_CRYPTO = (
    "OpenSSL libcrypto backs hashlib; its socket BIOs are never used (ssl/_ssl are "
    "excluded from the bundle and libssl is left out unless FFmpeg uses it)"
)
_WHY_KERBEROS = (
    "MIT Kerberos/GSSAPI, linked by QtNetwork for HTTP Negotiate authentication; the "
    "app makes no HTTP requests (no QNetworkAccessManager, enforced by the source check)"
)
_WHY_X11 = (
    "X11/XCB client libraries: a TCP connection is only made when the user's $DISPLAY "
    "names another host"
)
_WHY_X11_SESSION = (
    "X session management (libSM/libICE, linked by Qt's xcb platform plugin): ICE "
    "connects to the session manager named by $SESSION_MANAGER, a local socket on "
    "desktops; libSM resolves the local host name only to build its client ID"
)

#: Every bundled binary that may link networking code, and why that is acceptable.
#: Entries are per file and per indicator, so a new networking dependency of an
#: allowed library still fails the check until someone reviews it here.
BUNDLE_ALLOWLIST: tuple[AllowRule, ...] = (
    # CPython runtime and standard-library extension modules.
    _allow("_socket*.pyd", {"ws2_32", "iphlpapi"}, _WHY_PYTHON),
    _allow("_socket*.so", _RESOLVER, _WHY_PYTHON),
    _allow("select*.pyd", {"ws2_32"}, "select() on Windows is part of Winsock"),
    _allow("_overlapped*.pyd", {"ws2_32"}, "asyncio's Windows event loop (stdlib import)"),
    _allow("_multiprocessing*.pyd", {"ws2_32"}, "multiprocessing's C helpers (stdlib import)"),
    _allow("python3*.dll", {"ws2_32"}, _WHY_PYTHON),
    _allow("libpython3*", _RESOLVER, _WHY_PYTHON),
    _allow("python", _RESOLVER, _WHY_PYTHON),  # macOS framework build ("Python")
    _allow("libcrypto*", {"ws2_32", *_RESOLVER}, _WHY_CRYPTO),
    # psutil.
    _allow(
        "psutil/_psutil_*",
        {"iphlpapi", "ws2_32", *_RESOLVER},
        "psutil reads local network-interface statistics; it opens no connections",
    ),
    # Qt.
    _allow(
        "qt6core.dll",
        {"ws2_32", "mpr", "netapi32"},
        "QtCore: Winsock for its event dispatcher, MPR/NETAPI32 to resolve network "
        "drive letters in file paths; it makes no remote connections",
    ),
    # PySide6's QtNetwork bindings link Qt's network library; nothing else may.
    _allow("pyside6/qtnetwork.*", {"qt6network", "libqt6network", "qtnetwork"}, _WHY_QT_NETWORK),
    _allow("qt6network.dll", {"ws2_32", "iphlpapi", "dnsapi", "winhttp"}, _WHY_QT_NETWORK),
    _allow(
        "libqt6network.so*",
        {"libgssapi_krb5", "libresolv", "res_n*", "__res_n*", *_RESOLVER},
        _WHY_QT_NETWORK,
    ),
    _allow(
        "qtnetwork.framework/versions/*/qtnetwork",
        {"cfnetwork", "network", "gss", "libresolv", *_RESOLVER},
        _WHY_QT_NETWORK,
    ),
    _allow("libxcb.so*", _RESOLVER, _WHY_X11),
    _allow("libx11.so*", _RESOLVER, _WHY_X11),
    _allow("libsm.so*", _RESOLVER, _WHY_X11_SESSION),
    _allow("libice.so*", _RESOLVER, _WHY_X11_SESSION),
    _allow(
        "libdbus-1.so*",
        _RESOLVER,
        "D-Bus client (Qt's DBus module): the session bus is a local socket",
    ),
    _allow(
        "libsystemd.so*",
        _RESOLVER,
        "systemd client library, pulled in by libdbus; only its local APIs are used",
    ),
    # MIT Kerberos, which QtNetwork links for HTTP "Negotiate" authentication.
    _allow("libgssapi_krb5.so*", {"libkrb5", "libkrb5support", *_RESOLVER}, _WHY_KERBEROS),
    _allow(
        "libkrb5.so*",
        {"libkrb5support", "libresolv", "res_n*", "__res_n*", *_RESOLVER},
        _WHY_KERBEROS,
    ),
    _allow("libkrb5support.so*", _RESOLVER, _WHY_KERBEROS),
    _allow("libk5crypto.so*", {"libkrb5support"}, _WHY_KERBEROS),
    # pyobjc (macOS).
    _allow("objc/_objc.*.so", {"getaddrinfo", "getnameinfo"}, _WHY_PYOBJC),
    _allow("foundation/_foundation.*.so", {"getnameinfo"}, _WHY_PYOBJC_NETSERVICE),
    # OpenCV.
    _allow("opencv_videoio_ffmpeg*.dll", {"ws2_32"}, _WHY_FFMPEG),
    _allow(
        "libavformat*",
        {"libssl*", "libcrypto*", "libgnutls*", "libnghttp2*", *_RESOLVER},
        _WHY_FFMPEG,
    ),
    # OpenCV's macOS wheel: what Homebrew's FFmpeg brings along (see _WHY_FFMPEG_PROTOCOL).
    _allow("cv2/*dylibs/libavformat.*.dylib", {"libzmq"}, _WHY_FFMPEG_PROTOCOL),
    _allow(
        "cv2/*dylibs/libavdevice.*.dylib", {"libcurl", "libgnutls", "libzmq"}, _WHY_FFMPEG_LINKS
    ),
    _allow(
        "cv2/*dylibs/libavfilter.*.dylib", {"libcurl", "libgnutls", "libzmq"}, _WHY_FFMPEG_LINKS
    ),
    _allow("cv2/*dylibs/librist.*.dylib", {"getaddrinfo"}, _WHY_FFMPEG_PROTOCOL),
    _allow(
        "cv2/*dylibs/libsrt.*.dylib", {"libssl", "getaddrinfo", "getnameinfo"}, _WHY_FFMPEG_PROTOCOL
    ),
    _allow(
        "cv2/*dylibs/libssh.*.dylib",
        {"kerberos", "getaddrinfo", "getnameinfo"},  # Kerberos: GSSAPI logins
        _WHY_FFMPEG_PROTOCOL,
    ),
    _allow("cv2/*dylibs/libzmq.*.dylib", {"getaddrinfo", "getnameinfo"}, _WHY_FFMPEG_PROTOCOL),
    _allow("cv2/*dylibs/libtesseract.*.dylib", {"libcurl", "getaddrinfo"}, _WHY_TESSERACT),
    _allow("cv2/*dylibs/libxcb.*.dylib", {"getaddrinfo"}, _WHY_XCB_MACOS),
)


@dataclass(frozen=True)
class NativeBinary:
    """What a native binary imports: libraries and (ELF/Mach-O) undefined symbols."""

    format: str  # "pe", "elf" or "macho"
    libraries: tuple[str, ...]
    symbols: frozenset[str]


class BinaryFormatError(ValueError):
    """A file looked like a native binary but its headers could not be parsed."""


@dataclass(frozen=True)
class Indicator:
    """A networking library or symbol that a native binary imports."""

    rule: str  # "network-library" or "network-symbol"
    name: str  # as written in the binary ("WS2_32.dll", "getaddrinfo")
    key: str  # normalised for the allow-list ("ws2_32", "getaddrinfo")


@dataclass(frozen=True)
class BundleFinding:
    """A privacy-relevant finding in one bundled file."""

    relative: str
    rule: str
    detail: str
    allowed_by: AllowRule | None = None

    def __str__(self) -> str:
        return f"{self.relative}: {self.rule}: {self.detail}"


@dataclass
class BundleResult:
    """Outcome of scanning a frozen bundle."""

    files: int = 0
    binaries: int = 0
    #: Distinct Python modules and scripts read from PyInstaller archives and ZIP files.
    python_modules: int = 0
    violations: list[BundleFinding] = field(default_factory=list)
    allowed: list[BundleFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


# ----------------------------------------------------------------- binary readers
def _cstring(data: bytes, offset: int, limit: int = 4096) -> str:
    if not 0 <= offset < len(data):
        raise BinaryFormatError(f"string offset {offset:#x} outside the file")
    end = data.find(b"\0", offset, offset + limit)
    if end < 0:
        raise BinaryFormatError(f"unterminated string at {offset:#x}")
    return data[offset:end].decode("utf-8", "replace")


def binary_format(head: bytes) -> str | None:
    """``"pe"``, ``"elf"``, ``"macho"`` from the first bytes of a file, else ``None``."""
    if head[:4] == b"\x7fELF":
        return "elf"
    if head[:2] == b"MZ":
        return "pe"
    if head[:4] in (
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
    ):
        return "macho"
    # Universal binaries share CAFEBABE with Java class files; the architecture
    # count is small, whereas a class file has its (>= 45) version number there.
    if head[:4] in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf") and len(head) >= 8:
        (count,) = struct.unpack_from(">I", head, 4)
        if 0 < count < 20:
            return "macho"
    return None


def parse_pe(data: bytes) -> NativeBinary:
    """DLL names imported by a PE file (normal and delay-load import tables)."""
    try:
        return _parse_pe(data)
    except (struct.error, IndexError) as exc:
        raise BinaryFormatError(f"truncated PE file: {exc}") from exc


def _parse_pe(data: bytes) -> NativeBinary:
    (pe_offset,) = struct.unpack_from("<I", data, 0x3C)
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise BinaryFormatError("MZ file without a PE header")
    coff = pe_offset + 4
    (section_count,) = struct.unpack_from("<H", data, coff + 2)
    (optional_size,) = struct.unpack_from("<H", data, coff + 16)
    optional = coff + 20
    (magic,) = struct.unpack_from("<H", data, optional)
    if magic == 0x10B:  # PE32
        (image_base,) = struct.unpack_from("<I", data, optional + 28)
        directories = optional + 96
    elif magic == 0x20B:  # PE32+
        (image_base,) = struct.unpack_from("<Q", data, optional + 24)
        directories = optional + 112
    else:
        raise BinaryFormatError(f"unknown PE optional header magic {magic:#x}")
    (directory_count,) = struct.unpack_from("<I", data, directories - 4)

    sections: list[tuple[int, int, int, int]] = []
    table = optional + optional_size
    for index in range(section_count):
        virtual_size, address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, table + 40 * index + 8
        )
        sections.append((address, max(virtual_size, raw_size), raw_offset, raw_size))

    def offset_of(rva: int) -> int:
        for address, size, raw_offset, raw_size in sections:
            if address <= rva < address + size:
                if rva - address >= raw_size:
                    break
                return raw_offset + rva - address
        if sections and rva < min(s[0] for s in sections):
            return rva  # inside the headers
        raise BinaryFormatError(f"RVA {rva:#x} is not in any section")

    def directory(index: int) -> tuple[int, int]:
        if index >= directory_count:
            return 0, 0
        rva, size = struct.unpack_from("<II", data, directories + 8 * index)
        return int(rva), int(size)

    libraries: list[str] = []
    import_rva, _size = directory(1)
    if import_rva:
        position = offset_of(import_rva)
        for _ in range(4096):
            descriptor = struct.unpack_from("<IIIII", data, position)
            if not any(descriptor):
                break
            libraries.append(_cstring(data, offset_of(descriptor[3])))
            position += 20
    delay_rva, _size = directory(13)
    if delay_rva:
        position = offset_of(delay_rva)
        for _ in range(4096):
            attributes, name = struct.unpack_from("<II", data, position)
            if not (attributes or name):
                break
            # Old (VC6) descriptors hold virtual addresses instead of RVAs.
            rva = name if attributes & 1 else name - image_base
            libraries.append(_cstring(data, offset_of(rva)))
            position += 32
    return NativeBinary("pe", tuple(libraries), frozenset())


def parse_elf(data: bytes) -> NativeBinary:
    """``DT_NEEDED`` libraries and undefined dynamic symbols of an ELF file."""
    try:
        return _parse_elf(data)
    except (struct.error, IndexError) as exc:
        raise BinaryFormatError(f"truncated ELF file: {exc}") from exc


def _parse_elf(data: bytes) -> NativeBinary:
    elf_class, encoding = data[4], data[5]
    if elf_class not in (1, 2) or encoding not in (1, 2):
        raise BinaryFormatError("unknown ELF class or byte order")
    is64 = elf_class == 2
    end = "<" if encoding == 1 else ">"
    if is64:
        section_offset = struct.unpack_from(end + "Q", data, 0x28)[0]
        entry_size, count = struct.unpack_from(end + "HH", data, 0x3A)
    else:
        section_offset = struct.unpack_from(end + "I", data, 0x20)[0]
        entry_size, count = struct.unpack_from(end + "HH", data, 0x2E)

    # (type, offset, size, link) of every section.
    sections: list[tuple[int, int, int, int]] = []
    for index in range(count):
        base = section_offset + index * entry_size
        if is64:
            kind = struct.unpack_from(end + "I", data, base + 4)[0]
            offset, size = struct.unpack_from(end + "QQ", data, base + 24)
            link = struct.unpack_from(end + "I", data, base + 40)[0]
        else:
            kind = struct.unpack_from(end + "I", data, base + 4)[0]
            offset, size, link = struct.unpack_from(end + "III", data, base + 16)
        sections.append((kind, offset, size, link))

    libraries: list[str] = []
    symbols: set[str] = set()
    for kind, offset, size, link in sections:
        if kind == 6 and link < len(sections):  # SHT_DYNAMIC
            strings = sections[link][1]
            step = 16 if is64 else 8
            fmt = end + ("qQ" if is64 else "iI")
            for position in range(offset, offset + size, step):
                tag, value = struct.unpack_from(fmt, data, position)
                if tag == 0:
                    break
                if tag == 1:  # DT_NEEDED
                    libraries.append(_cstring(data, strings + value))
        elif kind == 11 and link < len(sections):  # SHT_DYNSYM
            strings = sections[link][1]
            step = 24 if is64 else 16
            for position in range(offset + step, offset + size, step):  # entry 0 is null
                if is64:
                    name, info, _other, index = struct.unpack_from(end + "IBBH", data, position)
                else:
                    name = struct.unpack_from(end + "I", data, position)[0]
                    info, _other, index = struct.unpack_from(end + "BBH", data, position + 12)
                if name and index == 0 and info >> 4 in (1, 2):  # undefined GLOBAL/WEAK
                    symbols.add(_cstring(data, strings + name))
    return NativeBinary("elf", tuple(libraries), frozenset(symbols))


_LC_SYMTAB = 0x2
_LC_DYLIB_COMMANDS = frozenset({0xC, 0x20, 0x80000018, 0x8000001F, 0x80000023})


def parse_macho(data: bytes) -> NativeBinary:
    """Linked dylibs/frameworks and undefined symbols of a (universal) Mach-O file."""
    try:
        return _parse_macho(data)
    except (struct.error, IndexError) as exc:
        raise BinaryFormatError(f"truncated Mach-O file: {exc}") from exc


def _parse_macho(data: bytes) -> NativeBinary:
    if data[:4] in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
        wide = data[3] == 0xBF
        (count,) = struct.unpack_from(">I", data, 4)
        libraries: list[str] = []
        symbols: set[str] = set()
        for index in range(count):
            if wide:
                offset, size = struct.unpack_from(">QQ", data, 8 + 32 * index + 8)
            else:
                offset, size = struct.unpack_from(">II", data, 8 + 20 * index + 8)
            thin = _parse_thin_macho(data[offset : offset + size])
            libraries += [lib for lib in thin.libraries if lib not in libraries]
            symbols |= thin.symbols
        return NativeBinary("macho", tuple(libraries), frozenset(symbols))
    return _parse_thin_macho(data)


def _parse_thin_macho(data: bytes) -> NativeBinary:
    magic = data[:4]
    if magic in (b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe"):
        end = "<"
    elif magic in (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf"):
        end = ">"
    else:
        raise BinaryFormatError("not a Mach-O slice")
    is64 = magic in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf")
    (command_count,) = struct.unpack_from(end + "I", data, 16)
    position = 32 if is64 else 28
    libraries: list[str] = []
    symbols: set[str] = set()
    for _ in range(command_count):
        command, size = struct.unpack_from(end + "II", data, position)
        if size < 8:
            raise BinaryFormatError("corrupt Mach-O load command")
        if command in _LC_DYLIB_COMMANDS:
            (name_offset,) = struct.unpack_from(end + "I", data, position + 8)
            libraries.append(_cstring(data, position + name_offset))
        elif command == _LC_SYMTAB:
            symbol_offset, symbol_count, string_offset, _string_size = struct.unpack_from(
                end + "IIII", data, position + 8
            )
            step = 16 if is64 else 12
            for index in range(symbol_count):
                name, kind = struct.unpack_from(end + "IB", data, symbol_offset + index * step)
                # Undefined external, not a debugging (stab) entry.
                if name and kind & 0xE0 == 0 and kind & 0x0E == 0 and kind & 0x01:
                    symbol = _cstring(data, string_offset + name)
                    symbols.add(symbol[1:] if symbol.startswith("_") else symbol)
        position += size
    return NativeBinary("macho", tuple(libraries), frozenset(symbols))


_PARSERS: dict[str, Callable[[bytes], NativeBinary]] = {
    "pe": parse_pe,
    "elf": parse_elf,
    "macho": parse_macho,
}


def parse_binary(data: bytes) -> NativeBinary | None:
    """Parse a native binary; ``None`` if ``data`` is not one."""
    kind = binary_format(data[:8])
    return None if kind is None else _PARSERS[kind](data)


# ------------------------------------------------------------------ bundle scanning
_MARKERS_LOWER: tuple[tuple[str, bytes, bytes], ...] = tuple(
    (marker, marker.lower().encode("ascii"), marker.lower().encode("utf-16-le"))
    for marker in TELEMETRY_MARKERS
)


def telemetry_markers(data: bytes) -> list[str]:
    """Telemetry markers found in ``data``, case-insensitively, as ASCII or UTF-16LE."""
    # bytes.lower() folds only ASCII letters, which is right for both encodings,
    # and plain substring search is much faster than a regex over large libraries.
    lowered = data.lower()
    return [
        marker for marker, narrow, wide in _MARKERS_LOWER if narrow in lowered or wide in lowered
    ]


def _library_key(library: str) -> str:
    """Normalise a linked library name for the allow-list.

    ``WS2_32.dll`` → ``ws2_32``, ``libssl.so.3`` → ``libssl``,
    ``/usr/lib/libresolv.9.dylib`` → ``libresolv``, ``.../CFNetwork`` → ``cfnetwork``.
    """
    name = library.replace("\\", "/").rsplit("/", 1)[-1].lower()
    name = name.removesuffix(".dll").removesuffix(".dylib").split(".so", 1)[0]
    return re.sub(r"(\.\d+)+$", "", name)


def _is_network_framework(library: str) -> bool:
    path = library.replace("\\", "/").lower()
    return any(f"/{name.lower()}.framework/" in path for name in MACOS_NETWORK_FRAMEWORKS)


def network_indicators(binary: NativeBinary) -> list[Indicator]:
    """Every networking library or symbol a native binary imports."""
    found: list[Indicator] = []
    for library in binary.libraries:
        key = _library_key(library)
        if key in QT_NETWORK_LIBRARY_KEYS:
            flagged = True
        elif binary.format == "pe":
            flagged = key in WINDOWS_NETWORK_DLLS
        else:
            flagged = key.startswith(POSIX_NETWORK_LIBRARIES) or _is_network_framework(library)
        if flagged:
            found.append(Indicator("network-library", library, key))
    found += [
        Indicator("network-symbol", symbol, symbol.lower())
        for symbol in sorted(binary.symbols & NETWORK_SYMBOLS)
    ]
    return found


def forbidden_distribution(name: str) -> str | None:
    """The forbidden distribution a bundle folder belongs to (``mediapipe-1.0.dist-info``)."""
    lowered = name.lower()
    if lowered.endswith(".dist-info"):
        lowered = lowered.removesuffix(".dist-info").rsplit("-", 1)[0]
    distribution = re.sub(r"[-_.]+", "-", lowered)
    return distribution if distribution in FORBIDDEN_DISTRIBUTIONS else None


def forbidden_module(module: str) -> str | None:
    """The forbidden package a bundled Python module belongs to, or ``None``.

    ``requests.adapters`` → ``requests``, ``sentry_sdk.client`` → ``sentry_sdk``;
    a vendored copy counts too (``somepkg._vendor.urllib3.util`` →
    ``somepkg._vendor.urllib3``). Modules of the standard library are never
    reported (see the module docstring).
    """
    parts = module.split(".")
    if not parts[0] or parts[0] in sys.stdlib_module_names:
        return None
    for index, part in enumerate(parts):
        if part in THIRD_PARTY_NETWORK_MODULES or forbidden_distribution(part):
            return ".".join(parts[: index + 1])
    return next(
        (sub for sub in NETWORK_SUBMODULES if module == sub or module.startswith(sub + ".")), None
    )


def network_plugin(relative: str) -> str | None:
    """Why a bundled file is a networking Qt plugin, or ``None`` if it is not one."""
    parts = PurePosixPath(relative.lower()).parts
    for index, part in enumerate(parts[:-2]):
        if part != "plugins":
            continue
        group, name = parts[index + 1], parts[-1]
        for plugin_group, pattern, reason in QT_NETWORK_PLUGINS:
            if group == plugin_group and fnmatch.fnmatchcase(name, pattern):
                return f"bundles the Qt plugin {group}/{name}: {reason}"
    return None


def _iter_bundle_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            # Symlinks (framework "Current" links, soname aliases) point at files
            # that are scanned themselves.
            if not path.is_symlink():
                yield path


# ------------------------------------------------------------- Python code in bundles
#: The cookie that ends the archive (CArchive/PKG) appended to a PyInstaller
#: executable: magic, archive length, TOC offset, TOC length, Python version,
#: Python library name. A PYZ archive starts with ``PYZ\0``.
_CARCHIVE_MAGIC = b"MEI\014\013\012\013\016"
_CARCHIVE_COOKIE = struct.Struct("!8sIIII64s")
_PYZ_MAGIC = b"PYZ\0"
_ZIP_MAGIC = b"PK\x03\x04"


class ArchiveError(ValueError):
    """Python code in the bundle that cannot be read, and so cannot be verified."""


@dataclass(frozen=True)
class EmbeddedItem:
    """Something stored inside a bundled file: Python code or an embedded file."""

    container: str  # "PYZ-00.pyz", "PKG" (the executable's archive) or "ZIP"
    name: str  # dotted module name, script name or file name
    kind: str  # "module", "script", "binary" (a file extracted at run time) or "data"
    data: bytes | None  # decompressed contents; None for a namespace package


def has_pyinstaller_archive(data: bytes) -> bool:
    """Whether ``data`` carries a PyInstaller archive (a well-formed cookie).

    The bootloader's code contains the magic bytes too, but only the last
    occurrence can be the cookie; code signatures appended after it (macOS) do
    not contain them.
    """
    offset = data.rfind(_CARCHIVE_MAGIC)
    if offset < 0 or offset + _CARCHIVE_COOKIE.size > len(data):
        return False
    _magic, length, toc_offset, toc_length, _python, library = _CARCHIVE_COOKIE.unpack_from(
        data, offset
    )
    end = offset + _CARCHIVE_COOKIE.size
    return bool(library.strip(b"\0")) and 0 < length <= end and toc_offset + toc_length <= length


def _archive_readers() -> tuple[Any, Any]:
    """PyInstaller's ``CArchiveReader`` and ``ZlibArchiveReader`` classes.

    The gate reads the archives with the reader of the PyInstaller that wrote
    them, so a change of the archive format cannot make it silently blind.
    """
    try:
        from PyInstaller.archive.readers import CArchiveReader
        from PyInstaller.loader.pyimod01_archive import ZlibArchiveReader
    except ImportError as exc:  # the release builds always have it (the "build" group)
        raise ArchiveError(
            f"PyInstaller is not installed, so the Python code cannot be read ({exc})"
        ) from exc
    return CArchiveReader, ZlibArchiveReader


def _pyz_items(pyz: Any, container: str) -> Iterator[EmbeddedItem]:
    for module in sorted(pyz.toc):
        # raw=True: the decompressed marshal data. String constants (URLs,
        # host names) appear in it verbatim, so no code object is loaded.
        yield EmbeddedItem(container, module, "module", pyz.extract(module, raw=True))


def read_pyinstaller_archive(path: Path) -> list[EmbeddedItem]:
    """Everything in the archive of a PyInstaller executable (or a PYZ file).

    Raises :class:`ArchiveError` when it cannot be read.
    """
    carchive_reader, zlib_reader = _archive_readers()
    try:
        with path.open("rb") as stream:
            if stream.read(len(_PYZ_MAGIC)) == _PYZ_MAGIC:
                return list(_pyz_items(zlib_reader(str(path), 0), path.name))
        archive = carchive_reader(str(path))
        items: list[EmbeddedItem] = []
        for name, entry in sorted(archive.toc.items()):
            typecode = entry[-1]
            if typecode == "z":  # the PYZ with every pure-Python module
                items += _pyz_items(archive.open_embedded_archive(name), name)
            elif typecode in {"m", "M"}:  # bootstrap modules
                items.append(EmbeddedItem("PKG", name, "module", archive.extract(name)))
            elif typecode == "s":  # the entry point and run-time hooks
                items.append(EmbeddedItem("PKG", name, "script", archive.extract(name)))
            elif typecode in {"b", "x", "l"}:  # files a one-file build extracts at run time
                items.append(EmbeddedItem("PKG", name, "binary", archive.extract(name)))
        return items
    # Any failure of the reader means the code cannot be verified. The PYZ reader
    # raises SystemExit (not an Exception) when the file vanishes mid-read.
    except (Exception, SystemExit) as exc:
        raise ArchiveError(f"PyInstaller archive cannot be read: {exc}") from exc


def read_zip(data: bytes) -> list[EmbeddedItem]:
    """The members of a ZIP file, decompressed (``base_library.zip``...).

    Raises :class:`ArchiveError` when it cannot be read.
    """
    items: list[EmbeddedItem] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                stem, _dot, suffix = info.filename.rpartition(".")
                if suffix in {"py", "pyc"} and stem:
                    module = stem.replace("/", ".").removesuffix(".__init__")
                    items.append(EmbeddedItem("ZIP", module, "module", archive.read(info)))
                else:
                    items.append(EmbeddedItem("ZIP", info.filename, "data", archive.read(info)))
    except Exception as exc:  # corrupt, encrypted or unsupported: cannot be verified
        raise ArchiveError(f"ZIP file cannot be read: {exc}") from exc
    return items


class _BundleScanner:
    """Collects the findings of one bundle scan."""

    def __init__(self, allowlist: Sequence[AllowRule]) -> None:
        self.allowlist = allowlist
        self.result = BundleResult()
        # Canonical names of forbidden packages already reported (once each).
        self._reported: set[str] = set()
        # The GUI and CLI executables embed the same modules: check each once.
        self._seen: set[bytes] = set()

    def add(self, finding: BundleFinding) -> None:
        (self.result.allowed if finding.allowed_by else self.result.violations).append(finding)

    def forbidden(self, relative: str, package: str, detail: str) -> None:
        key = re.sub(r"[-_.]+", "-", package.lower())
        if key not in self._reported:
            self._reported.add(key)
            self.add(BundleFinding(relative, "forbidden-package", detail))

    def scan_file(self, path: Path, relative: str) -> None:
        """One file of the bundle: its path, its bytes and any archive it carries."""
        self.result.files += 1
        for part in PurePosixPath(relative).parts[:-1]:
            distribution = forbidden_distribution(part)
            if distribution is not None:
                self.forbidden(
                    relative, distribution, f"bundles the '{distribution}' distribution ({part})"
                )
        plugin = network_plugin(relative)
        if plugin is not None:
            self.add(BundleFinding(relative, "network-plugin", plugin))
        try:
            data = path.read_bytes()
        except OSError as exc:
            self.add(BundleFinding(relative, "unreadable-binary", f"cannot be read: {exc}"))
            return
        self.scan_data(data, relative)
        if has_pyinstaller_archive(data) or data.startswith(_PYZ_MAGIC):
            self.scan_archive(relative, lambda: read_pyinstaller_archive(path))
        elif data.startswith(_ZIP_MAGIC):
            self.scan_archive(relative, lambda: read_zip(data))

    def scan_data(self, data: bytes, relative: str) -> None:
        findings, is_binary = scan_bundle_file(data, relative, self.allowlist)
        self.result.binaries += is_binary
        for finding in findings:
            self.add(finding)

    def scan_archive(self, relative: str, read: Callable[[], list[EmbeddedItem]]) -> None:
        """Check what ``read`` returns; an archive that cannot be read fails the gate."""
        try:
            items = read()
        except ArchiveError as exc:
            self.add(BundleFinding(relative, "unreadable-archive", f"cannot be verified: {exc}"))
            return
        for item in items:
            self.scan_item(relative, item)

    def scan_item(self, relative: str, item: EmbeddedItem) -> None:
        """One module, script or file from an archive inside ``relative``."""
        where = f"{item.container} in {relative}"
        if item.kind == "module":
            package = forbidden_module(item.name)
            if package is not None:
                self.forbidden(
                    relative,
                    package,
                    f"bundles the Python package '{package}' (module '{item.name}' in {where})",
                )
        data = item.data
        if data is None:
            return
        digest = hashlib.sha256(f"{item.kind}\0{item.name}\0".encode() + data).digest()
        if digest in self._seen:
            return
        self._seen.add(digest)
        if item.kind == "binary":
            # A file of a one-file build: checked like a bundled file, and a ZIP
            # among them (base_library.zip) is opened as well.
            nested = f"{relative}/{item.name}"
            self.scan_data(data, nested)
            if data.startswith(_ZIP_MAGIC):
                self.scan_archive(nested, lambda: read_zip(data))
            return
        if item.kind in {"module", "script"}:
            self.result.python_modules += 1
        for marker in telemetry_markers(data):
            detail = f"{item.kind} '{item.name}' ({where}) contains the telemetry marker '{marker}'"
            self.add(BundleFinding(relative, "telemetry-endpoint", detail))


def scan_bundle_file(
    data: bytes, relative: str, allowlist: Sequence[AllowRule] = BUNDLE_ALLOWLIST
) -> tuple[list[BundleFinding], bool]:
    """Findings for one bundled file, and whether it is a native binary."""
    findings = [
        BundleFinding(relative, "telemetry-endpoint", f"contains the telemetry marker '{marker}'")
        for marker in telemetry_markers(data)
    ]
    try:
        binary = parse_binary(data)
    except BinaryFormatError as exc:
        findings.append(BundleFinding(relative, "unreadable-binary", f"cannot be verified: {exc}"))
        return findings, True
    if binary is None:
        return findings, False
    for indicator in network_indicators(binary):
        rule = next((r for r in allowlist if r.matches(relative) and r.allows(indicator.key)), None)
        verb = "links" if indicator.rule == "network-library" else "imports"
        findings.append(BundleFinding(relative, indicator.rule, f"{verb} {indicator.name}", rule))
    return findings, True


def scan_bundle(root: Path, allowlist: Sequence[AllowRule] = BUNDLE_ALLOWLIST) -> BundleResult:
    """Scan a frozen bundle (a folder, a ``.app``, or a single file)."""
    scanner = _BundleScanner(allowlist)
    base = root.parent if root.is_file() else root
    for path in [root] if root.is_file() else _iter_bundle_files(root):
        scanner.scan_file(path, path.relative_to(base).as_posix())
    return scanner.result


# ------------------------------------------------------------------------ command line
def _github_annotation(violation: Violation) -> str:
    message = f"{violation.rule}: {violation.message}".replace("%", "%25").replace("\n", "%0A")
    return (
        f"::error file={_display_path(violation.path)},line={violation.line},"
        f"col={violation.col},title=Privacy check::{message}"
    )


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _allowed_summary(findings: Iterable[BundleFinding]) -> list[str]:
    """One line per allow-listed file: what it links and why that is accepted."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for finding in findings:
        reason = finding.allowed_by.reason if finding.allowed_by else ""
        what = finding.detail.split(" ", 1)[-1]  # "links WS2_32.dll" -> "WS2_32.dll"
        grouped.setdefault((finding.relative, reason), []).append(what)
    return [
        f"{relative}: allowed {', '.join(names)}: {reason}"
        for (relative, reason), names in grouped.items()
    ]


def _main_bundle(root: Path, *, quiet: bool, annotate: bool) -> int:
    result = scan_bundle(root)
    if not quiet:
        for line in _allowed_summary(result.allowed):
            print(f"::notice title=Privacy check (bundle)::{_escape(line)}" if annotate else line)
    for finding in result.violations:
        text = str(finding)
        print(f"::error title=Privacy check (bundle)::{_escape(text)}" if annotate else text)
    shown = _display_path(root)
    if result.ok:
        if not quiet:
            print(
                f"check_privacy: OK: {result.binaries} native binaries and "
                f"{result.python_modules} Python modules in {result.files} files of {shown}; "
                "no telemetry, and networking only where allow-listed."
            )
        return 0
    print(
        f"check_privacy: FAILED: {len(result.violations)} problem(s) in the bundle {shown}. "
        "Review each one; a library that legitimately links networking code needs an "
        "entry with a reason in BUNDLE_ALLOWLIST (scripts/check_privacy.py).",
        file=sys.stderr,
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if the source tree contains network or frame-writing code, or (--bundle) "
            "if a frozen build contains telemetry or unexpected networking libraries."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"files or directories to scan (default: {_display_path(DEFAULT_TARGET)})",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        metavar="DIR",
        help="scan a PyInstaller output folder (or .app) instead of Python sources",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="print violations only")
    args = parser.parse_args(argv)
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"

    if args.bundle is not None:
        if args.paths:
            parser.error("--bundle cannot be combined with source paths")
        if not args.bundle.exists():
            print(f"check_privacy: no such file or directory: {args.bundle}", file=sys.stderr)
            return 2
        return _main_bundle(args.bundle, quiet=args.quiet, annotate=annotate)

    targets: list[Path] = args.paths or [DEFAULT_TARGET]
    missing = [p for p in targets if not p.exists()]
    if missing:
        for path in missing:
            print(f"check_privacy: no such file or directory: {path}", file=sys.stderr)
        return 2

    result = scan_paths(targets)
    for violation in result.violations:
        print(_github_annotation(violation) if annotate else violation)

    shown = ", ".join(_display_path(p) for p in targets)
    if result.ok:
        if not args.quiet:
            print(
                f"check_privacy: OK: {len(result.files)} files in {shown}; "
                "no network or frame-writing code found."
            )
        return 0
    bad_files = len({v.path for v in result.violations})
    print(
        f"check_privacy: FAILED: {len(result.violations)} violation(s) in {bad_files} of "
        f"{len(result.files)} files. Eye Tracker must not use the network or write frames.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
