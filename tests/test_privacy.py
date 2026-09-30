"""Tests for scripts/check_privacy.py, the "no network, no frames on disk" guard."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_privacy.py"


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
    "source",
    [
        # Local IPC is the one allowed use of QtNetwork.
        "from PySide6.QtNetwork import QLocalServer, QLocalSocket, QAbstractSocket",
        "from PySide6 import QtNetwork\nQtNetwork.QLocalSocket()\nQtNetwork.QLocalServer.listen",
        "import PySide6.QtNetwork as qn\nqn.QLocalServer()",
        "from PySide6.QtNetwork import QLocalSocket\ns = QLocalSocket.LocalSocketState.Connected",
        "from PySide6 import QtNetwork\ngetattr(QtNetwork, 'QLocalSocket')",
        "from PySide6.QtWidgets import QApplication\nfrom PySide6.QtGui import QImage",
        # Requesting MJPG from a camera builds a codec code; nothing is written.
        "import cv2\ncode = cv2.VideoWriter.fourcc(*'MJPG')",
        "import cv2\ncode = cv2.VideoWriter_fourcc(*'MJPG')",
        "import cv2\nframe = cv2.imdecode(buf, cv2.IMREAD_COLOR)",
        # Ordinary system tools, local modules and look-alike names are fine.
        "import subprocess\nsubprocess.run(['loginctl', 'lock-session'], check=False)",
        "import subprocess\nsubprocess.run(['gdbus', 'call', '--session'])",
        "from . import socket\nfrom .http import thing",
        "import select, selectors, os, sys",
        "import sockets_helper\nimport httpish",
        "import importlib\nmodule = importlib.import_module(f'{__name__}.windows')",
        "import ctypes\nuser32 = ctypes.WinDLL('user32', use_last_error=True)",
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
