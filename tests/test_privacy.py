"""Tests for scripts/check_privacy.py, the "no network, no frames on disk" guard.

It checks the source tree and (``--bundle``) the frozen PyInstaller output; the
bundle tests use small synthetic PE, ELF and Mach-O files built here, and
PyInstaller archives written with PyInstaller's own writers.
"""

from __future__ import annotations

import importlib.util
import io
import os
import struct
import subprocess
import sys
import textwrap
import zipfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_privacy.py"

# A source distribution ships the tests but not scripts/: skip instead of
# failing collection there.
if not SCRIPT.is_file():
    pytest.skip("build tooling is not part of this source tree", allow_module_level=True)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_privacy", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_privacy = _load_script()


def _rules(source: str) -> list[str]:
    violations = check_privacy.scan_source(textwrap.dedent(source), Path("snippet.py"))
    return [v.rule for v in violations]


# ---------------------------------------------------------------------------- the real tree
def test_source_tree_is_clean() -> None:
    result = check_privacy.scan_paths([check_privacy.DEFAULT_TARGET])
    assert len(result.files) > 10, "the scan should cover the whole package"
    assert result.ok, "privacy violations:\n" + "\n".join(str(v) for v in result.violations)


def test_default_target_is_the_package() -> None:
    assert check_privacy.DEFAULT_TARGET == REPO_ROOT / "src" / "eye_tracker"
    assert (check_privacy.DEFAULT_TARGET / "__init__.py").is_file()


# --------------------------------------------------------------------------------- violations
@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("import socket", "network-import"),
        ("import urllib.request", "network-import"),
        ("from http import client", "network-import"),
        ("from http.server import HTTPServer", "network-import"),
        ("import requests as r", "network-import"),
        ("from aiohttp import ClientSession", "network-import"),
        ("import websockets", "network-import"),
        ("import ssl, os", "network-import"),
        ("def f():\n    import ftplib", "network-import"),
        ("import importlib\nimportlib.import_module('smtplib')", "network-import"),
        ("__import__('httpx')", "network-import"),
        ("import sentry_sdk", "network-import"),
        ("import mediapipe as mp", "network-import"),
        ("from PySide6.QtNetwork import QTcpSocket", "qt-network"),
        ("from PySide6.QtNetwork import QLocalSocket, QNetworkAccessManager", "qt-network"),
        ("from PySide6.QtNetwork import *", "qt-network"),
        ("from PySide6 import QtNetwork\nQtNetwork.QNetworkAccessManager()", "qt-network"),
        ("from PySide6 import QtNetwork as net\nnet.QUdpSocket()", "qt-network"),
        ("import PySide6.QtNetwork as qn\nqn.QTcpServer()", "qt-network"),
        ("import PySide6.QtNetwork\nPySide6.QtNetwork.QSslSocket()", "qt-network"),
        ("from PyQt6.QtNetwork import QTcpSocket", "qt-network"),
        ("from PySide6 import QtNetwork\ngetattr(QtNetwork, 'QTcpSocket')", "qt-network"),
        ("from PySide6.QtWebEngineWidgets import QWebEngineView", "qt-network"),
        ("from PySide6 import QtWebSockets", "qt-network"),
        ("import importlib\nimportlib.import_module('PySide6.QtNetwork')", "qt-network"),
        ("import cv2\ncv2.imwrite('face.png', frame)", "frame-write"),
        ("import cv2\nok, buf = cv2.imencode('.jpg', frame)", "frame-write"),
        ("import cv2\nw = cv2.VideoWriter('out.avi', 0, 15.0, (640, 480))", "frame-write"),
        ("from cv2 import imwrite", "frame-write"),
        ("import cv2\nsave = cv2.imwrite", "frame-write"),
        ("import ctypes\nctypes.WinDLL('ws2_32')", "native-network"),
        ("import ctypes\nctypes.CDLL('libcurl.so.4')", "native-network"),
        ("import ctypes\nctypes.windll.wininet.InternetOpenW", "native-network"),
        ("import subprocess\nsubprocess.run(['curl', '-O', 'https://x.org'])", "network-command"),
        ("import os\nos.system('wget http://example.org/x')", "network-command"),
        ("CMD = ('C:\\\\tools\\\\curl.exe', '--help')", "network-command"),
        ("import shutil\nshutil.which('ssh')", "network-command"),
        ("def broken(:\n    pass", "syntax-error"),
    ],
)
def test_detects_violation(source: str, rule: str) -> None:
    assert rule in _rules(source)


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        # QAbstractSocket is constructible and has connectToHost(): not local IPC.
        ("from PySide6.QtNetwork import QAbstractSocket", "qt-network"),
        (
            "from PySide6.QtNetwork import QAbstractSocket\n"
            "s = QAbstractSocket(QAbstractSocket.SocketType.TcpSocket, None)\n"
            "s.connectToHost('example.org', 443)",
            "qt-network",
        ),
        (
            "from PySide6 import QtNetwork\nQtNetwork.QAbstractSocket.SocketType.TcpSocket",
            "qt-network",
        ),
        # The class names give it away however the module was reached.
        ("import PySide6.QtNetwork as n\nm = n\nm.QTcpSocket()", "qt-network"),
        ("mod = load()\nmod.QNetworkAccessManager()", "qt-network"),
        ("getattr(load(), 'QUdpSocket')", "qt-network"),
        # asyncio streams, servers and loop connections.
        ("import asyncio\nasyncio.open_connection('example.org', 443)", "network-api"),
        ("import asyncio\nasyncio.start_server(handle, '0.0.0.0', 8080)", "network-api"),
        ("from asyncio import open_connection", "network-api"),
        ("loop.create_connection(Protocol, 'example.org', 443)", "network-api"),
        ("loop.create_datagram_endpoint(Protocol, remote_addr=('x', 53))", "network-api"),
        ("loop.create_server(Protocol, port=1)", "network-api"),
        ("await loop.sock_connect(sock, ('example.org', 80))", "network-api"),
        ("import asyncio\nasyncio.open_unix_connection('/tmp/s')", "network-api"),
        # multiprocessing's sockets.
        ("import multiprocessing.connection", "network-import"),
        ("from multiprocessing.connection import Client", "network-import"),
        ("from multiprocessing import connection", "network-import"),
        ("from multiprocessing.managers import BaseManager", "network-import"),
        (
            "import multiprocessing\nmultiprocessing.connection.Client(('example.org', 443))",
            "network-import",
        ),
        (
            "import importlib\nimportlib.import_module('multiprocessing.connection')",
            "network-import",
        ),
        # The C modules behind socket, ssl and asyncio/multiprocessing on Windows.
        ("import _socket\n_socket.socket().connect(('example.org', 80))", "network-import"),
        ("import _ssl", "network-import"),
        ("from _overlapped import WSAConnect", "network-import"),
        ("import _multiprocessing", "network-import"),
        # Raw sockets through ctypes (what select() would then poll).
        ("import ctypes\nlibc = ctypes.CDLL(None)\nfd = libc.socket(2, 1, 0)", "native-network"),
        ("import ctypes\nctypes.CDLL('libc.so.6').connect(fd, addr, 16)", "native-network"),
        (
            "import ctypes\n"
            "def _libc():\n    return ctypes.CDLL(None, use_errno=True)\n"
            "lib = _libc()\nlib.connect(fd, addr, 16)",
            "native-network",
        ),
        (
            "import ctypes\n"
            "class C:\n"
            "    def __init__(self):\n        self._libc = ctypes.CDLL(None)\n"
            "    def go(self):\n        self._libc.sendto(fd, b'x', 1, 0, addr, 16)",
            "native-network",
        ),
        ("import ctypes\nlibc = ctypes.CDLL(None)\ngetattr(libc, 'connect')", "native-network"),
        ("import ctypes\nctypes.cdll.msvcrt.bind(fd, addr, 16)", "native-network"),
        (
            "import ctypes\nlibc = ctypes.CDLL(None)\nlibc.getaddrinfo(host, None, None, res)",
            "native-network",
        ),
        ("import ctypes\nctypes.windll.ws2_32.WSAStartup(0x202, data)", "native-network"),
        ("api.WinHttpOpen(None, 0, None, None, 0)", "native-network"),
        ("getattr(lib, 'gethostbyname')", "native-network"),
        ("import ctypes\nctypes.WinDLL('winhttp.dll')", "native-network"),
        ("import ctypes\nctypes.WinDLL('dnsapi')", "native-network"),
        ("import ctypes\nctypes.CDLL('libcurl-gnutls.so.4')", "native-network"),
        ("import ctypes.util\nctypes.util.find_library('curl')", "native-network"),
        ("import ctypes\nctypes.CDLL('/usr/lib/x86_64-linux-gnu/libssl.so.3')", "native-network"),
        (
            "import ctypes\n"
            "ctypes.CDLL('/System/Library/Frameworks/CFNetwork.framework/CFNetwork')",
            "native-network",
        ),
        # Network tools behind shells and wrappers.
        (
            "import subprocess\nsubprocess.run(['sh', '-c', 'curl https://x.org'])",
            "network-command",
        ),
        ("import subprocess\nsubprocess.run(['sudo', 'wget', 'https://x.org'])", "network-command"),
        ("import os\nos.system('echo hi && curl https://x.org')", "network-command"),
        (
            "import subprocess\nsubprocess.run('powershell -c iwr https://x.org', shell=True)",
            "network-command",
        ),
        (
            "import subprocess\nsubprocess.run(['cmd', '/c', 'certutil -urlcache -f x y'])",
            "network-command",
        ),
        (
            "from PySide6.QtCore import QProcess\nQProcess.startDetached('curl', ['x'])",
            "network-command",
        ),
        ("import os\nos.posix_spawn('/usr/bin/wget', args, env)", "network-command"),
        # Log records sent over the network.
        ("import logging.handlers\nlogging.handlers.HTTPHandler('x.org', '/log')", "network-api"),
        ("from logging.handlers import SocketHandler", "network-api"),
        (
            "import logging.handlers\nh = logging.handlers.SysLogHandler(('x.org', 514))",
            "network-api",
        ),
        ("import logging.config\nlogging.config.listen(9999)", "network-api"),
    ],
)
def test_detects_bypasses(source: str, rule: str) -> None:
    """Ordinary APIs that used to slip past the check (packaging-07)."""
    assert rule in _rules(source)


@pytest.mark.parametrize(
    "source",
    [
        # Local IPC is the one allowed use of QtNetwork.
        "from PySide6.QtNetwork import QLocalServer, QLocalSocket",
        "from PySide6 import QtNetwork\nQtNetwork.QLocalSocket()\nQtNetwork.QLocalServer.listen",
        "import PySide6.QtNetwork as qn\nqn.QLocalServer()",
        "from PySide6.QtNetwork import QLocalSocket\ns = QLocalSocket.LocalSocketState.Connected",
        "from PySide6.QtNetwork import QLocalSocket\ns = QLocalSocket()\ns.connectToServer('x')",
        "from PySide6 import QtNetwork\ngetattr(QtNetwork, 'QLocalSocket')",
        "from PySide6.QtWidgets import QApplication\nfrom PySide6.QtGui import QImage",
        # Qt signals, servers of the local kind and look-alike method names.
        "button.clicked.connect(on_click)\nserver.listen(name)\nsocket.accept_later()",
        "self.worker.connect(self.slot)\nwatcher.bind(key)",
        # Requesting MJPG from a camera builds a codec code; nothing is written.
        "import cv2\ncode = cv2.VideoWriter.fourcc(*'MJPG')",
        "import cv2\ncode = cv2.VideoWriter_fourcc(*'MJPG')",
        "import cv2\nframe = cv2.imdecode(buf, cv2.IMREAD_COLOR)",
        # Ordinary system tools, local modules and look-alike names are fine.
        "import subprocess\nsubprocess.run(['loginctl', 'lock-session'], check=False)",
        "import subprocess\nsubprocess.run(['gdbus', 'call', '--session'])",
        "import subprocess\nsubprocess.run(['sh', '-c', 'xset dpms force off'])",
        "import subprocess\nsubprocess.run(['sudo', 'true'])",
        "import os\nos.system('xdg-open /tmp')",
        "from . import socket\nfrom .http import thing",
        "import select, selectors, os, sys",
        "import select\nselect.select([fd], [], [], 1.0)",
        "import sockets_helper\nimport httpish",
        "import importlib\nmodule = importlib.import_module(f'{__name__}.windows')",
        "import ctypes\nuser32 = ctypes.WinDLL('user32', use_last_error=True)",
        "import ctypes\nuser32 = ctypes.WinDLL('user32')\nuser32.GetCursorPos(point)",
        "import ctypes\nlibc = ctypes.CDLL(None)\nlibc.fanotify_init(0, 0)\nlibc.inotify_init1(0)",
        "import ctypes.util\nctypes.CDLL(ctypes.util.find_library('c'))",
        "import logging.handlers\nlogging.handlers.RotatingFileHandler('app.log')",
        "import multiprocessing\nmultiprocessing.cpu_count()",
        "import asyncio\nasyncio.run(main())\nasyncio.sleep(1)",
        "def create_menu():\n    pass",
        "REPO_URL = 'https://github.com/bugraskl/eye-tracker'",
        "ws = 'curl is just a word in a string'",
    ],
)
def test_allows_legitimate_code(source: str) -> None:
    assert _rules(source) == []


def test_reports_location(tmp_path: Path) -> None:
    bad = tmp_path / "leaky.py"
    bad.write_text("import os\n\nimport socket\n", encoding="utf-8")
    [violation] = check_privacy.scan_file(bad)
    assert (violation.path, violation.line, violation.col) == (bad, 3, 1)
    assert violation.rule == "network-import"
    assert str(violation).endswith("3:1: network-import: import of networking module 'socket'")


def test_synthetic_violating_package_is_detected(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "clean.py").write_text("import numpy as np\n", encoding="utf-8")
    (package / "sub" / "leak.py").write_text(
        textwrap.dedent(
            """\
            import cv2
            from PySide6.QtNetwork import QLocalServer, QTcpServer


            def save(frame):
                cv2.imwrite("frame.png", frame)
            """
        ),
        encoding="utf-8",
    )
    # Byte-code caches are never scanned.
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "junk.py").write_text("import socket\n", encoding="utf-8")

    result = check_privacy.scan_paths([package])

    assert len(result.files) == 3
    assert not result.ok
    found = {(v.path.name, v.line, v.rule) for v in result.violations}
    assert found == {("leak.py", 2, "qt-network"), ("leak.py", 6, "frame-write")}


def test_source_encoding_cookie_is_honoured(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.py"
    legacy.write_bytes(b"# -*- coding: latin-1 -*-\nNAME = '\xe7a'\nimport socket\n")
    assert [v.rule for v in check_privacy.scan_file(legacy)] == ["network-import"]


# ------------------------------------------------------------------------------ command line
def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "ok.py").write_text("from PySide6.QtNetwork import QLocalSocket\n", encoding="utf-8")
    leaky = tmp_path / "leaky"
    leaky.mkdir()
    (leaky / "bad.py").write_text("import urllib.request\n", encoding="utf-8")

    assert check_privacy.main([str(clean)]) == 0
    assert "OK: 1 files" in capsys.readouterr().out

    assert check_privacy.main([str(leaky)]) == 1
    captured = capsys.readouterr()
    assert "bad.py:1:1: network-import" in captured.out
    assert "FAILED: 1 violation(s)" in captured.err

    assert check_privacy.main([str(tmp_path / "missing")]) == 2


def test_github_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "bad.py").write_text("import ssl\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert check_privacy.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("::error file=")
    assert ",line=1,col=1,title=Privacy check::network-import:" in out


def test_script_runs_standalone(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("import socket\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--quiet", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 1, proc.stderr
    assert "network-import" in proc.stdout


# ================================================================ synthetic native binaries
def _pe(imports: Sequence[str] = (), delay: Sequence[str] = (), *, pe32: bool = False) -> bytes:
    """A minimal PE image with one section holding the import tables."""
    section_rva, section_offset = 0x1000, 0x200
    descriptors = 20 * (len(imports) + 1)
    delay_start = descriptors
    names_start = delay_start + 32 * (len(delay) + 1)
    names = b""
    name_rvas: list[int] = []
    for name in (*imports, *delay):
        name_rvas.append(section_rva + names_start + len(names))
        names += name.encode("ascii") + b"\0"
    body = b"".join(struct.pack("<IIIII", 0, 0, 0, rva, 0) for rva in name_rvas[: len(imports)])
    body += b"\0" * 20
    body += b"".join(
        struct.pack("<IIIIIIII", 1, rva, 0, 0, 0, 0, 0, 0) for rva in name_rvas[len(imports) :]
    )
    body += b"\0" * 32 + names
    body += b"\0" * (-len(body) % 0x200)

    optional_size = 224 if pe32 else 240
    optional = bytearray(optional_size)
    struct.pack_into("<H", optional, 0, 0x10B if pe32 else 0x20B)
    directories = 96 if pe32 else 112
    struct.pack_into("<I", optional, directories - 4, 16)
    struct.pack_into("<II", optional, directories + 8, section_rva, descriptors)
    if delay:
        struct.pack_into("<II", optional, directories + 8 * 13, section_rva + delay_start, 32)
    coff = struct.pack("<HHIIIHH", 0x14C if pe32 else 0x8664, 1, 0, 0, 0, optional_size, 0x2022)
    section = struct.pack(
        "<8sIIIIIIHHI", b".idata", len(body), section_rva, len(body), section_offset, 0, 0, 0, 0, 0
    )
    dos = bytearray(0x40)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x40)
    header = bytes(dos) + b"PE\0\0" + coff + bytes(optional) + section
    return header + b"\0" * (section_offset - len(header)) + body


def _elf(
    needed: Sequence[str] = (),
    undefined: Sequence[str] = (),
    defined: Sequence[str] = (),
    *,
    is64: bool = True,
    big_endian: bool = False,
) -> bytes:
    """A minimal ELF shared object: .dynstr, .dynamic and .dynsym sections only."""
    end = ">" if big_endian else "<"
    strings = b"\0"
    offsets: dict[str, int] = {}
    for name in (*needed, *undefined, *defined):
        offsets[name] = len(strings)
        strings += name.encode("ascii") + b"\0"
    word, entry = ("Q", 16) if is64 else ("I", 8)
    dynamic = b"".join(
        struct.pack(end + ("q" if is64 else "i") + word, 1, offsets[lib]) for lib in needed
    )
    dynamic += b"\0" * entry
    symbols = b"\0" * (24 if is64 else 16)  # the null symbol
    for name, index in [(n, 0) for n in undefined] + [(n, 7) for n in defined]:
        info = (1 << 4) | 2  # GLOBAL FUNC
        if is64:
            symbols += struct.pack(end + "IBBHQQ", offsets[name], info, 0, index, 0, 0)
        else:
            symbols += struct.pack(end + "IIIBBH", offsets[name], 0, 0, info, 0, index)
    header_size = 64 if is64 else 52
    strings_offset = header_size
    dynamic_offset = strings_offset + len(strings)
    symbols_offset = dynamic_offset + len(dynamic)
    sections_offset = symbols_offset + len(symbols)

    def section(kind: int, offset: int, size: int, link: int) -> bytes:
        if is64:
            return struct.pack(end + "IIQQQQIIQQ", 0, kind, 0, 0, offset, size, link, 0, 0, 0)
        return struct.pack(end + "IIIIIIIIII", 0, kind, 0, 0, offset, size, link, 0, 0, 0)

    table = (
        section(0, 0, 0, 0)
        + section(3, strings_offset, len(strings), 0)
        + section(6, dynamic_offset, len(dynamic), 1)
        + section(11, symbols_offset, len(symbols), 1)
    )
    ident = b"\x7fELF" + bytes([2 if is64 else 1, 2 if big_endian else 1, 1]) + b"\0" * 9
    if is64:
        header = ident + struct.pack(
            end + "HHIQQQIHHHHHH", 3, 62, 1, 0, 0, sections_offset, 0, 64, 56, 0, 64, 4, 0
        )
    else:
        header = ident + struct.pack(
            end + "HHIIIIIHHHHHH", 3, 3, 1, 0, 0, sections_offset, 0, 52, 32, 0, 40, 4, 0
        )
    return header + strings + dynamic + symbols + table


def _macho(
    dylibs: Sequence[str] = (),
    undefined: Sequence[str] = (),
    defined: Sequence[str] = (),
    *,
    fat: bool = False,
) -> bytes:
    """A minimal 64-bit little-endian Mach-O dylib, optionally in a universal wrapper."""
    commands = b""
    for path in dylibs:
        name = path.encode("utf-8") + b"\0"
        name += b"\0" * (-(24 + len(name)) % 8)
        commands += struct.pack("<IIIIII", 0xC, 24 + len(name), 24, 0, 0, 0) + name
    strings = b"\0"
    entries: list[tuple[int, int]] = []
    for names, kind in ((undefined, 0x01), (defined, 0x0F)):
        for symbol in names:
            entries.append((len(strings), kind))
            strings += b"_" + symbol.encode("ascii") + b"\0"
    header_size = 32
    commands_size = len(commands) + 24
    symbols_offset = header_size + commands_size
    symbols = b"".join(struct.pack("<IBBHQ", offset, kind, 0, 0, 0) for offset, kind in entries)
    strings_offset = symbols_offset + len(symbols)
    commands += struct.pack(
        "<IIIIII", 0x2, 24, symbols_offset, len(entries), strings_offset, len(strings)
    )
    count = len(dylibs) + 1
    header = struct.pack("<IiiIIIII", 0xFEEDFACF, 0x0100000C, 0, 6, count, commands_size, 0, 0)
    thin = header + commands + symbols + strings
    if not fat:
        return thin
    offset = 0x1000
    wrapper = struct.pack(">II", 0xCAFEBABE, 1) + struct.pack(
        ">iiIII", 0x0100000C, 0, offset, len(thin), 12
    )
    return wrapper + b"\0" * (offset - len(wrapper)) + thin


# ------------------------------------------------------------------------- binary readers
@pytest.mark.parametrize("pe32", [False, True])
def test_pe_imports_and_delay_imports(pe32: bool) -> None:
    data = _pe(["KERNEL32.dll", "WS2_32.dll"], delay=["WININET.dll"], pe32=pe32)
    binary = check_privacy.parse_binary(data)
    assert binary.format == "pe"
    assert binary.libraries == ("KERNEL32.dll", "WS2_32.dll", "WININET.dll")
    keys = [indicator.key for indicator in check_privacy.network_indicators(binary)]
    assert keys == ["ws2_32", "wininet"]


@pytest.mark.parametrize(("is64", "big_endian"), [(True, False), (False, True), (False, False)])
def test_elf_needed_libraries_and_undefined_symbols(is64: bool, big_endian: bool) -> None:
    data = _elf(
        needed=["libc.so.6", "libcurl.so.4"],
        undefined=["getaddrinfo", "connect", "socket"],
        defined=["gethostbyname"],  # defined here, not imported: not an indicator
        is64=is64,
        big_endian=big_endian,
    )
    binary = check_privacy.parse_binary(data)
    assert binary.format == "elf"
    assert binary.libraries == ("libc.so.6", "libcurl.so.4")
    assert binary.symbols == {"getaddrinfo", "connect", "socket"}
    indicators = check_privacy.network_indicators(binary)
    # socket/connect are also local IPC: only the resolver and libcurl count.
    assert [(i.rule, i.key) for i in indicators] == [
        ("network-library", "libcurl"),
        ("network-symbol", "getaddrinfo"),
    ]


@pytest.mark.parametrize("fat", [False, True])
def test_macho_dylibs_and_undefined_symbols(fat: bool) -> None:
    data = _macho(
        dylibs=[
            "/System/Library/Frameworks/CFNetwork.framework/Versions/A/CFNetwork",
            "/usr/lib/libSystem.B.dylib",
            "/usr/lib/libresolv.9.dylib",
        ],
        undefined=["getaddrinfo", "malloc"],
        defined=["gethostbyname"],
        fat=fat,
    )
    binary = check_privacy.parse_binary(data)
    assert binary.format == "macho"
    assert binary.libraries[1] == "/usr/lib/libSystem.B.dylib"
    assert binary.symbols == {"getaddrinfo", "malloc"}
    keys = [i.key for i in check_privacy.network_indicators(binary)]
    assert keys == ["cfnetwork", "libresolv", "getaddrinfo"]


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"plain text, not a binary",
        b"\x89PNG\r\n\x1a\n" + b"\0" * 64,
        # A Java class file shares the universal-binary magic number.
        b"\xca\xfe\xba\xbe\x00\x00\x00\x34" + b"\0" * 64,
    ],
)
def test_non_binaries_are_not_parsed(data: bytes) -> None:
    assert check_privacy.parse_binary(data) is None


@pytest.mark.parametrize(
    "data",
    [
        _pe(["WS2_32.dll"])[:0x90],
        b"MZ" + b"\0" * 0x3A + struct.pack("<I", 0x40) + b"NOPE" + b"\0" * 64,
        _elf(["libc.so.6"], ["getaddrinfo"])[:80],
        _macho(["/usr/lib/libSystem.B.dylib"])[:40],
    ],
)
def test_truncated_binaries_are_reported(data: bytes) -> None:
    with pytest.raises(check_privacy.BinaryFormatError):
        check_privacy.parse_binary(data)


def test_telemetry_markers_ascii_and_utf16_any_case() -> None:
    assert check_privacy.telemetry_markers(b"xx https://PLAY.googleapis.com/log xx") == [
        "play.googleapis.com"
    ]
    wide = "AVClearcutLogger".encode("utf-16-le")
    assert check_privacy.telemetry_markers(b"\0\0" + wide) == ["clearcut"]
    # The model NOTICE names where the weights came from; that is not telemetry.
    assert check_privacy.telemetry_markers(b"https://storage.googleapis.com/mediapipe-models") == []


def test_committed_models_contain_no_telemetry_markers() -> None:
    models = REPO_ROOT / "src" / "eye_tracker" / "vision" / "models"
    files = [p for p in models.iterdir() if p.is_file()]
    assert files
    for path in files:
        assert check_privacy.telemetry_markers(path.read_bytes()) == [], path


def test_allowlist_entries_are_explained_and_normalised() -> None:
    for rule in check_privacy.BUNDLE_ALLOWLIST:
        assert rule.reason.strip(), rule
        assert rule.indicators, rule
        assert all(i == i.lower() for i in rule.indicators), rule
    # Markers are never allow-listed: that list only holds libraries and symbols.
    every = set().union(*(r.indicators for r in check_privacy.BUNDLE_ALLOWLIST))
    assert not every & {m.lower() for m in check_privacy.TELEMETRY_MARKERS}


# ------------------------------------------------------------------------------ bundle scans
def _write(root: Path, relative: str, data: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _bundle(tmp_path: Path, files: dict[str, bytes]) -> Path:
    root = tmp_path / "EyeTracker"
    for relative, data in files.items():
        _write(root, relative, data)
    return root


def _summary(result: object) -> set[tuple[str, str]]:
    return {(f.relative, f.rule) for f in result.violations}  # type: ignore[attr-defined]


def test_clean_bundle_passes_with_allowed_networking(tmp_path: Path) -> None:
    root = _bundle(
        tmp_path,
        {
            "EyeTracker.exe": _pe(["KERNEL32.dll"]),
            "_internal/_socket.pyd": _pe(["WS2_32.dll", "IPHLPAPI.DLL", "python312.dll"]),
            "_internal/PySide6/Qt6Network.dll": _pe(["WS2_32.dll", "DNSAPI.dll", "WINHTTP.dll"]),
            "_internal/PySide6/Qt/lib/libQt6Network.so.6": _elf(
                ["libgssapi_krb5.so.2", "libc.so.6"], ["getaddrinfo", "res_nquery"]
            ),
            "_internal/libxcb.so.1": _elf(["libXau.so.6"], ["getaddrinfo", "connect"]),
            "_internal/cv2/opencv_python_headless.libs/libavformat-4d3f2b1c.so.61.1.100": _elf(
                ["libssl-8e5a0a1b.so.3"], ["getaddrinfo"]
            ),
            "Contents/Frameworks/PySide6/Qt/lib/QtNetwork.framework/Versions/A/QtNetwork": _macho(
                ["/System/Library/Frameworks/CFNetwork.framework/Versions/A/CFNetwork"],
                ["getaddrinfo"],
            ),
            "_internal/eye_tracker/vision/models/NOTICE.md": (
                b"https://storage.googleapis.com/mediapipe-models/face_landmarker.task"
            ),
            "_internal/numpy-2.5.3.dist-info/METADATA": b"Name: numpy",
        },
    )
    result = check_privacy.scan_bundle(root)
    assert result.ok, [str(v) for v in result.violations]
    assert result.binaries == 7
    assert result.files == 9
    allowed = {(f.relative.rsplit("/", 1)[-1], f.detail) for f in result.allowed}
    assert ("Qt6Network.dll", "links WINHTTP.dll") in allowed
    assert ("libQt6Network.so.6", "imports getaddrinfo") in allowed


@pytest.mark.parametrize(
    ("relative", "needed", "undefined"),
    [
        # What these libraries import on Ubuntu (measured with readelf / nm -D).
        ("_internal/libxcb.so.1", ["libXau.so.6", "libXdmcp.so.6"], ["getaddrinfo"]),
        ("_internal/libX11.so.6", ["libxcb.so.1"], ["getaddrinfo"]),
        ("_internal/libdbus-1.so.3", ["libsystemd.so.0"], ["getaddrinfo", "getnameinfo"]),
        ("_internal/libsystemd.so.0", ["libcap.so.2"], ["getaddrinfo"]),
        ("_internal/libcrypto.so.3", [], ["getaddrinfo", "gethostbyname", "getnameinfo"]),
        ("_internal/libgssapi_krb5.so.2", ["libkrb5.so.3", "libkrb5support.so.0"], []),
        (
            "_internal/libkrb5.so.3",
            ["libkrb5support.so.0", "libresolv.so.2"],
            ["getnameinfo", "res_nsearch"],
        ),
        ("_internal/libkrb5support.so.0", [], ["getaddrinfo", "getnameinfo"]),
        ("_internal/libk5crypto.so.3", ["libkrb5support.so.0"], []),
        (
            "_internal/lib-dynload/_socket.cpython-312-x86_64-linux-gnu.so",
            [],
            ["getaddrinfo", "gethostbyaddr_r", "gethostbyname_r", "getnameinfo"],
        ),
        ("_internal/libpython3.12.so.1.0", [], ["getaddrinfo", "getnameinfo"]),
        # X session management, linked by Qt's xcb plugin (xtrans and sm_genid use
        # the resolver; not measured on a runner, so the whole resolver is covered).
        ("_internal/libSM.so.6", ["libICE.so.6", "libuuid.so.1"], ["getaddrinfo", "gethostbyname"]),
        ("_internal/libICE.so.6", [], ["getaddrinfo", "getnameinfo", "gethostbyname_r"]),
        # PySide6's QtNetwork bindings, on Linux, Windows and macOS.
        ("_internal/PySide6/QtNetwork.abi3.so", ["libQt6Network.so.6", "libQt6Core.so.6"], []),
    ],
)
def test_linux_system_libraries_are_allowed(
    tmp_path: Path, relative: str, needed: list[str], undefined: list[str]
) -> None:
    root = _bundle(tmp_path, {relative: _elf(needed, [*undefined, "malloc"])})
    result = check_privacy.scan_bundle(root)
    assert result.ok, [str(v) for v in result.violations]


_MAC = "Contents/Frameworks/"
_MAC_CV2 = _MAC + "cv2/__dot__dylibs/"
_SYSTEM = "/System/Library/Frameworks/"


@pytest.mark.parametrize(
    ("relative", "dylibs", "undefined"),
    [
        # What the macOS bundle's libraries link and import (measured on the macos-14
        # runner with otool -L and nm -m: OpenCV 5.0.0.93, PySide6 6.11.2, pyobjc 12.2.2).
        (
            _MAC_CV2 + "libavformat.61.7.100.dylib",
            ["@rpath/libzmq.5.dylib", "@rpath/libgnutls.30.dylib", "@rpath/libsrt.1.5.4.dylib"],
            ["getaddrinfo", "getnameinfo"],
        ),
        (
            _MAC_CV2 + "libavdevice.61.3.100.dylib",
            ["/usr/lib/libcurl.4.dylib", "@rpath/libzmq.5.dylib", "@rpath/libgnutls.30.dylib"],
            [],
        ),
        (
            _MAC_CV2 + "libavfilter.10.4.100.dylib",
            ["/usr/lib/libcurl.4.dylib", "@rpath/libzmq.5.dylib", "@rpath/libgnutls.30.dylib"],
            [],
        ),
        (_MAC_CV2 + "librist.4.dylib", ["@rpath/libmbedcrypto.3.6.3.dylib"], ["getaddrinfo"]),
        (
            _MAC_CV2 + "libsrt.1.5.4.dylib",
            ["@rpath/libssl.3.dylib", "@rpath/libcrypto.3.dylib"],
            ["getaddrinfo", "getnameinfo"],
        ),
        (
            _MAC_CV2 + "libssh.4.10.1.dylib",
            ["@rpath/libcrypto.3.dylib", _SYSTEM + "Kerberos.framework/Versions/A/Kerberos"],
            ["getaddrinfo", "getnameinfo"],
        ),
        (
            _MAC_CV2 + "libzmq.5.dylib",
            ["@rpath/libsodium.26.dylib"],
            ["getaddrinfo", "getnameinfo"],
        ),
        (_MAC_CV2 + "libtesseract.5.dylib", ["/usr/lib/libcurl.4.dylib"], ["getaddrinfo"]),
        (_MAC_CV2 + "libxcb.1.1.0.dylib", ["@rpath/libXau.6.dylib"], ["getaddrinfo"]),
        (_MAC_CV2 + "libcrypto.3.dylib", [], ["getaddrinfo", "gethostbyname", "getnameinfo"]),
        (_MAC + "objc/_objc.cpython-312-darwin.so", [], ["getaddrinfo", "getnameinfo"]),
        (_MAC + "Foundation/_Foundation.cpython-312-darwin.so", [], ["getnameinfo"]),
        (
            _MAC + "PySide6/Qt/lib/QtNetwork.framework/Versions/A/QtNetwork",
            [
                _SYSTEM + "CFNetwork.framework/Versions/A/CFNetwork",
                _SYSTEM + "Network.framework/Versions/A/Network",
                _SYSTEM + "GSS.framework/Versions/A/GSS",
                "/usr/lib/libresolv.9.dylib",
            ],
            ["getaddrinfo", "getnameinfo"],
        ),
        (
            _MAC + "PySide6/QtNetwork.abi3.so",
            ["@rpath/QtNetwork.framework/Versions/A/QtNetwork"],
            [],
        ),
        (_MAC + "psutil/_psutil_osx.abi3.so", [], ["getnameinfo"]),
        (
            _MAC + "python3__dot__12/lib-dynload/_socket.cpython-312-darwin.so",
            [],
            ["getaddrinfo", "gethostbyaddr", "gethostbyname", "getnameinfo"],
        ),
    ],
)
def test_macos_bundle_libraries_are_allowed(
    tmp_path: Path, relative: str, dylibs: list[str], undefined: list[str]
) -> None:
    binary = _macho([*dylibs, "/usr/lib/libSystem.B.dylib"], [*undefined, "malloc"])
    result = check_privacy.scan_bundle(_bundle(tmp_path, {relative: binary}))
    assert result.ok, [str(v) for v in result.violations]
    assert result.allowed
    assert all(finding.allowed_by and finding.allowed_by.reason for finding in result.allowed)


def test_macos_allowlist_is_per_file_and_per_indicator(tmp_path: Path) -> None:
    """The macOS entries cover OpenCV's own copies of these libraries and only what
    they were measured to link; libX11, which nothing uses, is left out by the spec
    rather than allowed."""
    network = _SYSTEM + "Network.framework/Versions/A/Network"
    root = _bundle(
        tmp_path,
        {
            _MAC_CV2 + "libX11.6.dylib": _macho([], ["getaddrinfo"]),
            _MAC + "libzmq.5.dylib": _macho([], ["getaddrinfo"]),
            _MAC_CV2 + "libavdevice.61.3.100.dylib": _macho(["@rpath/libssl.3.dylib"]),
            _MAC_CV2 + "libtesseract.5.dylib": _macho([], ["gethostbyname"]),
            _MAC + "objc/_objc.cpython-312-darwin.so": _macho([], ["gethostbyname"]),
            _MAC + "Foundation/_Foundation.cpython-312-darwin.so": _macho([], ["getaddrinfo"]),
            _MAC + "PySide6/Qt/lib/QtGui.framework/Versions/A/QtGui": _macho([network]),
        },
    )
    result = check_privacy.scan_bundle(root)
    assert {(f.relative.rsplit("/", 1)[-1], f.detail) for f in result.violations} == {
        ("libX11.6.dylib", "imports getaddrinfo"),
        ("libzmq.5.dylib", "imports getaddrinfo"),
        ("libavdevice.61.3.100.dylib", "links @rpath/libssl.3.dylib"),
        ("libtesseract.5.dylib", "imports gethostbyname"),
        ("_objc.cpython-312-darwin.so", "imports gethostbyname"),
        ("_Foundation.cpython-312-darwin.so", "imports getaddrinfo"),
        ("QtGui", f"links {network}"),
    }
    assert result.allowed == []


@pytest.mark.parametrize("framework", ["CFNetwork", "GSS", "Kerberos", "LDAP", "Network"])
def test_macos_network_frameworks_are_networking_libraries(framework: str) -> None:
    """Apple's network frameworks, and the counterparts of libgssapi, libkrb5 and
    libldap, which the gate reports on Linux."""
    path = f"{_SYSTEM}{framework}.framework/Versions/A/{framework}"
    binary = check_privacy.parse_binary(_macho([path, _SYSTEM + "Security.framework/Security"]))
    assert binary is not None
    indicators = check_privacy.network_indicators(binary)
    assert [(i.rule, i.name, i.key) for i in indicators] == [
        ("network-library", path, framework.lower())
    ]


def test_gio_fails_the_gate(tmp_path: Path) -> None:
    """r2-packaging-01: GLib's GIO (pulled in by the GTK3 platform theme) resolves
    host names; the spec keeps it out of the bundle instead of allowing it."""
    gio = _elf(["libglib-2.0.so.0"], ["getaddrinfo", "getnameinfo", "res_nquery", "malloc"])
    result = check_privacy.scan_bundle(_bundle(tmp_path, {"_internal/libgio-2.0.so.0": gio}))
    assert {f.detail for f in result.violations} == {
        "imports getaddrinfo",
        "imports getnameinfo",
        "imports res_nquery",
    }


def test_qt_network_plugins_and_qtnetwork_users_fail(tmp_path: Path) -> None:
    """r2-packaging-05: Qt's VNC platform plugin (a TCP server), TLS backends and the
    TUIO listener fail by path, and anything but the QtNetwork bindings that links
    Qt's network library fails too."""
    root = _bundle(
        tmp_path,
        {
            "_internal/PySide6/Qt/plugins/platforms/libqvnc.so": _elf(
                ["libQt6Network.so.6", "libQt6Gui.so.6"]
            ),
            "_internal/PySide6/plugins/tls/qschannelbackend.dll": _pe(["Qt6Network.dll"]),
            "_internal/PySide6/Qt/plugins/generic/libqtuiotouchplugin.so": _elf(
                ["libQt6Network.so.6"]
            ),
            "Contents/Frameworks/PySide6/Qt/plugins/networkinformation/libqscnetworkreachability"
            ".dylib": _macho(["@rpath/QtNetwork.framework/Versions/A/QtNetwork"]),
            # Not plugins of a networking kind, but they gained a QtNetwork dependency.
            "_internal/PySide6/Qt/lib/libQt6Foo.so.6": _elf(["libQt6Network.so.6"]),
            # Fine: the bindings the app uses for QLocalServer/QLocalSocket...
            "_internal/PySide6/QtNetwork.pyd": _pe(["Qt6Network.dll", "Qt6Core.dll"]),
            "Contents/Frameworks/PySide6/QtNetwork.abi3.so": _macho(
                ["@rpath/QtNetwork.framework/Versions/A/QtNetwork"]
            ),
            # ...and plugins that do not touch the network.
            "_internal/PySide6/Qt/plugins/platforms/libqxcb.so": _elf(["libQt6XcbQpa.so.6"]),
            "_internal/PySide6/plugins/platforms/qwindows.dll": _pe(["Qt6Gui.dll"]),
            "_internal/PySide6/Qt/plugins/generic/libqevdevmouseplugin.so": _elf(["libudev.so.1"]),
        },
    )
    result = check_privacy.scan_bundle(root)
    found = {(f.relative.rsplit("/", 1)[-1], f.rule) for f in result.violations}
    assert found == {
        ("libqvnc.so", "network-plugin"),
        ("libqvnc.so", "network-library"),
        ("qschannelbackend.dll", "network-plugin"),
        ("qschannelbackend.dll", "network-library"),
        ("libqtuiotouchplugin.so", "network-plugin"),
        ("libqtuiotouchplugin.so", "network-library"),
        ("libqscnetworkreachability.dylib", "network-plugin"),
        ("libqscnetworkreachability.dylib", "network-library"),
        ("libQt6Foo.so.6", "network-library"),
    }
    vnc = next(f for f in result.violations if f.rule == "network-plugin" and "vnc" in f.relative)
    assert "unauthenticated TCP server" in vnc.detail
    assert {f.relative.rsplit("/", 1)[-1] for f in result.allowed} == {
        "QtNetwork.pyd",
        "QtNetwork.abi3.so",
    }


@pytest.mark.parametrize(
    ("relative", "reason"),
    [
        ("_internal/PySide6/Qt/plugins/platforms/libqvnc.so", "VNC"),
        ("_internal/PySide6/Qt/plugins/platforms/libqwebgl.so", "WebGL"),
        ("_internal/PySide6/plugins/TLS/qopensslbackend.dll", "TLS"),
        ("_internal/PySide6/plugins/networkaccess/qnetworkaccessbackend.dll", "QNetworkAccess"),
        ("_internal/PySide6/Qt/plugins/generic/libqtuiotouchplugin.so", "UDP"),
        ("_internal/PySide6/Qt/plugins/platforms/libqxcb.so", None),
        ("_internal/PySide6/Qt/plugins/platformthemes/libqxdgdesktopportal.so", None),
        ("_internal/PySide6/Qt/plugins/generic/libqevdevkeyboardplugin.so", None),
        ("_internal/vnc/plugins.txt", None),  # needs a plugin group below "plugins"
        ("_internal/libqvnc.so", None),
    ],
)
def test_network_plugin(relative: str, reason: str | None) -> None:
    detail = check_privacy.network_plugin(relative)
    if reason is None:
        assert detail is None
    else:
        assert detail is not None
        assert reason in detail


@pytest.mark.parametrize(
    ("module", "package"),
    [
        ("requests", "requests"),
        ("requests.adapters", "requests"),
        ("sentry_sdk.client", "sentry_sdk"),
        ("mediapipe.tasks.python", "mediapipe"),
        ("somepkg._vendor.urllib3.util", "somepkg._vendor.urllib3"),
        ("posthog", "posthog"),
        ("google.cloud.storage", "google.cloud"),
        # The standard library is not reported: it imports these itself.
        ("http.client", None),
        ("urllib.request", None),
        ("socket", None),
        ("multiprocessing.connection", None),
        # What the bundle legitimately contains.
        ("eye_tracker.ipc", None),
        ("PySide6.QtNetwork", None),
        ("cv2.dnn", None),
        ("psutil._pslinux", None),
        ("platformdirs.windows", None),
        ("", None),
    ],
)
def test_forbidden_module(module: str, package: str | None) -> None:
    assert check_privacy.forbidden_module(module) == package


def test_package_source_contains_no_telemetry_markers() -> None:
    """The app's own modules go into the PYZ, whose code the gate now reads: a
    docstring naming a telemetry endpoint would fail the release, so fail here first."""
    sources = sorted((REPO_ROOT / "src" / "eye_tracker").rglob("*.py"))
    assert sources
    for path in sources:
        assert check_privacy.telemetry_markers(path.read_bytes()) == [], path


# ----------------------------------------------------------------- Python code in bundles
def _writers() -> ModuleType:
    """PyInstaller's archive writers (the "build" dependency group)."""
    return pytest.importorskip("PyInstaller.archive.writers")


def _pyz(path: Path, modules: dict[str, str | None]) -> Path:
    """A PYZ archive written by PyInstaller: module name -> source (None: namespace package)."""
    entries: list[tuple[str, str, str]] = []
    code = {}
    for name, source in modules.items():
        if source is None:
            entries.append((name, "-", "PYMODULE"))
            continue
        filename = name.replace(".", "/") + ".py"
        code[name] = compile(source, filename, "exec")
        entries.append((name, filename, "PYMODULE"))
    _writers().ZlibArchiveWriter(str(path), entries, code_dict=code)
    return path


def _executable(
    tmp_path: Path, pyz: Path, extra: Sequence[tuple[str, str, bool, str]] = ()
) -> bytes:
    """A PyInstaller-style executable: a PE "bootloader" with a CArchive appended."""
    script = tmp_path / "entry.py"
    script.write_text("import eye_tracker.app\n", encoding="utf-8")
    package = tmp_path / "archive.pkg"
    entries = [("PYZ.pyz", str(pyz), False, "z"), ("entry", str(script), True, "s"), *extra]
    _writers().CArchiveWriter(str(package), entries, "python312.dll")
    return _pe(["KERNEL32.dll"]) + package.read_bytes()


_MODULES: dict[str, str | None] = {
    "eye_tracker.app": "TITLE = 'Eye Tracker'\n",
    "http.client": "PORT = 80\n",  # standard library: expected, not reported
    "requests": "",
    "requests.adapters": "",
    "somelib.stats": "URL = 'https://api.segment.io/v1/track'\n",
    "nspkg": None,
}


def test_python_code_inside_the_executables_is_checked(tmp_path: Path) -> None:
    """r2-packaging-03: the PYZ hides third-party modules and their strings from a
    byte scan; the gate reads it with PyInstaller's own reader."""
    exe = _executable(tmp_path, _pyz(tmp_path / "PYZ.pyz", _MODULES))
    # Compressed: invisible to the plain byte scan.
    assert check_privacy.telemetry_markers(exe) == []
    assert check_privacy.has_pyinstaller_archive(exe)
    # The GUI and the CLI executable embed the same archive.
    root = _bundle(tmp_path / "dist", {"EyeTracker.exe": exe, "eye-tracker-cli.exe": exe})

    result = check_privacy.scan_bundle(root)

    found = {(f.relative, f.rule, f.detail) for f in result.violations}
    assert found == {
        (
            "EyeTracker.exe",
            "forbidden-package",
            "bundles the Python package 'requests' (module 'requests' in PYZ.pyz in "
            "EyeTracker.exe)",
        ),
        (
            "EyeTracker.exe",
            "telemetry-endpoint",
            "module 'somelib.stats' (PYZ.pyz in EyeTracker.exe) contains the telemetry "
            "marker 'api.segment.io'",
        ),
    }
    # Five modules with code and the entry script, each counted once.
    assert result.python_modules == 6
    assert result.binaries == 2


def test_one_file_builds_have_their_embedded_files_checked(tmp_path: Path) -> None:
    library = tmp_path / "libfetch.dll"
    library.write_bytes(_pe(["WININET.dll"]))
    base_library = tmp_path / "base_library.zip"
    with zipfile.ZipFile(base_library, "w") as archive:
        archive.writestr("urllib3/__init__.pyc", b"\0" * 16)
    exe = _executable(
        tmp_path,
        _pyz(tmp_path / "PYZ.pyz", {"eye_tracker.app": "X = 1\n"}),
        [
            ("libfetch.dll", str(library), True, "b"),
            ("base_library.zip", str(base_library), True, "x"),
        ],
    )
    result = check_privacy.scan_bundle(_bundle(tmp_path / "dist", {"EyeTracker.exe": exe}))
    assert _summary(result) == {
        ("EyeTracker.exe/libfetch.dll", "network-library"),
        ("EyeTracker.exe/base_library.zip", "forbidden-package"),
    }


def test_zip_files_and_standalone_pyz_archives_are_checked(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("http/client.pyc", b"\0" * 16)
        archive.writestr("encodings/__init__.pyc", b"\0" * 16)
        archive.writestr("urllib3/__init__.pyc", b"\0" * 16)
        archive.writestr("certs/endpoints.txt", b"https://o1.ingest.sentry.io/api/1/")
    pyz = _pyz(tmp_path / "PYZ-00.pyz", {"posthog.client": "", "eye_tracker.ui": ""})
    root = _bundle(
        tmp_path / "dist",
        {"_internal/base_library.zip": buffer.getvalue(), "_internal/PYZ-00.pyz": pyz.read_bytes()},
    )
    result = check_privacy.scan_bundle(root)
    found = {(f.relative, f.rule, f.detail) for f in result.violations}
    assert found == {
        (
            "_internal/PYZ-00.pyz",
            "forbidden-package",
            "bundles the Python package 'posthog' (module 'posthog.client' in PYZ-00.pyz in "
            "_internal/PYZ-00.pyz)",
        ),
        (
            "_internal/base_library.zip",
            "forbidden-package",
            "bundles the Python package 'urllib3' (module 'urllib3' in ZIP in "
            "_internal/base_library.zip)",
        ),
        (
            "_internal/base_library.zip",
            "telemetry-endpoint",
            "data 'certs/endpoints.txt' (ZIP in _internal/base_library.zip) contains the "
            "telemetry marker 'ingest.sentry.io'",
        ),
    }
    assert result.python_modules == 5


def _cookie(
    archive_length: int, toc_offset: int, toc_length: int, library: bytes = b"python312.dll"
) -> bytes:
    """The trailer of a PyInstaller archive (see ``_CARCHIVE_COOKIE``)."""
    return struct.pack(
        "!8sIIII64s",
        b"MEI\014\013\012\013\016",
        archive_length,
        toc_offset,
        toc_length,
        312,
        library,
    )


def test_unreadable_archives_fail_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A well-formed cookie in front of a table of contents that does not parse.
    archive = b"\0" * 32 + b"\xff" * 16
    corrupt_exe = _pe(["KERNEL32.dll"]) + archive + _cookie(len(archive) + 88, 32, 16)
    assert check_privacy.has_pyinstaller_archive(corrupt_exe)
    root = _bundle(
        tmp_path / "dist",
        {
            "EyeTracker.exe": corrupt_exe,
            "_internal/base_library.zip": b"PK\x03\x04 not really a zip file",
        },
    )
    result = check_privacy.scan_bundle(root)
    assert _summary(result) == {
        ("EyeTracker.exe", "unreadable-archive"),
        ("_internal/base_library.zip", "unreadable-archive"),
    }

    # Without PyInstaller the Python code cannot be read: that fails too.
    good = _executable(tmp_path, _pyz(tmp_path / "PYZ.pyz", {"eye_tracker.app": ""}))
    monkeypatch.setitem(sys.modules, "PyInstaller.archive.readers", None)
    result = check_privacy.scan_bundle(_bundle(tmp_path / "nopyi", {"EyeTracker.exe": good}))
    [finding] = result.violations
    assert finding.rule == "unreadable-archive"
    assert "PyInstaller is not installed" in finding.detail


@pytest.mark.parametrize(
    "data",
    [
        _pe(["KERNEL32.dll"]),
        # The bootloader's own copy of the magic, without a cookie after it.
        _pe(["KERNEL32.dll"]) + b"MEI\014\013\012\013\016" + b"\0" * 8,
        # A cookie without a Python library name.
        _pe(["KERNEL32.dll"]) + b"\0" * 40 + _cookie(128, 0, 16, library=b""),
        # A cookie whose archive would start before the file does.
        b"MEI\014\013\012\013\016" + _cookie(10_000, 0, 16)[8:],
    ],
)
def test_files_without_a_pyinstaller_archive(data: bytes) -> None:
    assert not check_privacy.has_pyinstaller_archive(data)


def test_mediapipe_like_bundle_fails(tmp_path: Path) -> None:
    """What the MediaPipe runtime put into the 0.1 bundle must never pass again."""
    libmediapipe = _pe(["KERNEL32.dll", "WININET.dll"]) + b"https://play.googleapis.com/log\0"
    root = _bundle(
        tmp_path,
        {
            "_internal/mediapipe/tasks/c/libmediapipe.dll": libmediapipe,
            "_internal/mediapipe-1.0.1.dist-info/METADATA": b"Name: mediapipe",
        },
    )
    result = check_privacy.scan_bundle(root)
    assert not result.ok
    assert _summary(result) == {
        ("_internal/mediapipe/tasks/c/libmediapipe.dll", "forbidden-package"),
        ("_internal/mediapipe/tasks/c/libmediapipe.dll", "telemetry-endpoint"),
        ("_internal/mediapipe/tasks/c/libmediapipe.dll", "network-library"),
    }
    # One report per distribution, however many of its files are bundled.
    assert sum(v.rule == "forbidden-package" for v in result.violations) == 1


def test_unknown_or_unexpected_networking_fails(tmp_path: Path) -> None:
    root = _bundle(
        tmp_path,
        {
            # An allowed file that gains a new networking dependency.
            "_internal/_socket.pyd": _pe(["WS2_32.dll", "WINHTTP.dll"]),
            # Networking where none is expected.
            "_internal/somelib.dll": _pe(["WS2_32.dll"]),
            "_internal/libsomething.so.1": _elf(["libc.so.6"], ["gethostbyname"]),
            "_internal/libfetch.so": _elf(["libcurl-gnutls.so.4"]),
            "_internal/broken.dll": b"MZ" + b"\0" * 0x3A + struct.pack("<I", 0x40) + b"XXXX",
            "_internal/data/config.json": b'{"endpoint": "https://api.segment.io/v1"}',
        },
    )
    result = check_privacy.scan_bundle(root)
    details = {(f.relative.rsplit("/", 1)[-1], f.rule, f.detail) for f in result.violations}
    assert details == {
        ("_socket.pyd", "network-library", "links WINHTTP.dll"),
        ("somelib.dll", "network-library", "links WS2_32.dll"),
        ("libsomething.so.1", "network-symbol", "imports gethostbyname"),
        ("libfetch.so", "network-library", "links libcurl-gnutls.so.4"),
        ("broken.dll", "unreadable-binary", "cannot be verified: MZ file without a PE header"),
        ("config.json", "telemetry-endpoint", "contains the telemetry marker 'api.segment.io'"),
    }
    # The allowed part of _socket.pyd is still reported as allowed.
    assert [f.detail for f in result.allowed] == ["links WS2_32.dll"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_symlinks_are_not_followed(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"Versions/A/QtCore": _pe(["KERNEL32.dll"])})
    outside = tmp_path / "outside.dll"
    outside.write_bytes(_pe(["WININET.dll"]))
    os.symlink(outside, root / "linked.dll")
    os.symlink("A", root / "Versions" / "Current")
    result = check_privacy.scan_bundle(root)
    assert result.ok
    assert result.files == 1


def test_bundle_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _bundle(tmp_path / "good", {"_internal/_socket.pyd": _pe(["WS2_32.dll"])})
    bad = _bundle(tmp_path / "bad", {"_internal/x.dll": _pe(["WININET.dll"])})

    assert check_privacy.main(["--bundle", str(good)]) == 0
    out = capsys.readouterr().out
    assert "_internal/_socket.pyd: allowed WS2_32.dll: CPython" in out
    assert "OK: 1 native binaries and 0 Python modules in 1 files" in out

    assert check_privacy.main(["--bundle", str(bad), "--quiet"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "_internal/x.dll: network-library: links WININET.dll\n"
    assert "FAILED: 1 problem(s)" in captured.err

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert check_privacy.main(["--bundle", str(bad)]) == 1
    assert capsys.readouterr().out.startswith(
        "::error title=Privacy check (bundle)::_internal/x.dll"
    )

    assert check_privacy.main(["--bundle", str(tmp_path / "missing")]) == 2
    with pytest.raises(SystemExit) as exc:
        check_privacy.main(["--bundle", str(good), "src"])
    assert exc.value.code == 2


def test_bundle_scan_of_a_single_file(tmp_path: Path) -> None:
    library = tmp_path / "libmediapipe.so"
    library.write_bytes(_elf(["libc.so.6"]) + b"clearcut")
    result = check_privacy.scan_bundle(library)
    assert _summary(result) == {("libmediapipe.so", "telemetry-endpoint")}
