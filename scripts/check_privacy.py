#!/usr/bin/env python3
"""Static privacy check for the Eye Tracker source tree.

Eye Tracker promises that it makes no network connections and never writes
camera frames to disk. This script enforces that promise mechanically. It parses
(never imports) every Python file under ``src/eye_tracker`` and reports:

``network-import``
    An import of a networking module (``socket``, ``ssl``, ``http``, ``urllib``,
    ``requests``, ``httpx``, ``aiohttp``, ``websockets``, ``ftplib``, ...),
    including dynamic imports with a literal name (``importlib.import_module``).
``qt-network``
    A Qt network class other than the local IPC ones (``QLocalServer``,
    ``QLocalSocket`` and the ``QAbstractSocket`` enums), a wildcard import from
    ``QtNetwork``, or any Qt web/remote module (``QtWebSockets``, ``QtWebEngine*``...).
``frame-write``
    Any use of an image/video encoder that could put a frame on disk or into a
    byte buffer (``imwrite``, ``imencode``, ``VideoWriter``).
``native-network``
    A native networking library loaded through ctypes (``ws2_32``, ``wininet``...).
``network-command``
    A network command-line tool started as a subprocess (``curl``, ``wget``, ``ssh``...).
``syntax-error``
    A file that cannot be parsed, and therefore cannot be verified.

Usage::

    python scripts/check_privacy.py            # scan src/eye_tracker
    python scripts/check_privacy.py PATH ...   # scan other files or directories

Exit status: 0 when clean, 1 when violations were found, 2 on a usage error.

The check is deliberately conservative: it has no allow-list comments. If it
flags something legitimate, change the rule here so the exception is reviewed.
It cannot see through values computed at runtime (a module name built from
variables, ``QImage.save`` on a camera frame); code review covers those.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "src" / "eye_tracker"

#: Top-level modules that exist to talk to the network.
NETWORK_MODULES: frozenset[str] = frozenset(
    {
        "aiohttp",
        "asyncssh",
        "ftplib",
        "grpc",
        "http",
        "httplib2",
        "httpx",
        "imaplib",
        "nntplib",
        "paramiko",
        "poplib",
        "pycurl",
        "requests",
        "smtpd",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "telnetlib",
        "urllib",
        "urllib3",
        "websocket",
        "websockets",
        "wsgiref",
        "xmlrpc",
        "zmq",
    }
)

QT_BINDINGS: tuple[str, ...] = ("PySide6", "PySide2", "PyQt6", "PyQt5")

#: The only QtNetwork classes the app may use: local (same-machine) IPC.
QT_NETWORK_ALLOWED: frozenset[str] = frozenset({"QLocalServer", "QLocalSocket", "QAbstractSocket"})

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

#: Encoders that turn a frame into a file or a byte buffer.
FRAME_WRITE_NAMES: frozenset[str] = frozenset(
    {"imwrite", "imwritemulti", "imencode", "VideoWriter"}
)

#: Native libraries (Windows DLL stems / POSIX sonames) that implement networking.
NATIVE_NETWORK_LIBS: frozenset[str] = frozenset(
    {"ws2_32", "wsock32", "mswsock", "wininet", "winhttp", "urlmon", "libcurl", "curl"}
)

#: Command-line tools that transfer data over the network.
NETWORK_COMMANDS: frozenset[str] = frozenset(
    {
        "aria2c",
        "bitsadmin",
        "curl",
        "ftp",
        "invoke-restmethod",
        "invoke-webrequest",
        "irm",
        "iwr",
        "nc",
        "ncat",
        "netcat",
        "scp",
        "sftp",
        "ssh",
        "telnet",
        "wget",
    }
)

#: Call names whose first argument is a command line.
_COMMAND_CALLS: frozenset[str] = frozenset(
    {
        "Popen",
        "call",
        "check_call",
        "check_output",
        "create_subprocess_exec",
        "create_subprocess_shell",
        "execl",
        "execlp",
        "execute",
        "execv",
        "execvp",
        "getoutput",
        "getstatusoutput",
        "popen",
        "run",
        "spawnl",
        "spawnlp",
        "spawnv",
        "spawnvp",
        "start",
        "startDetached",
        "startfile",
        "system",
        "which",
    }
)

_LIBRARY_LOADERS: frozenset[str] = frozenset(
    {"CDLL", "OleDLL", "PyDLL", "WinDLL", "LoadLibrary", "find_library"}
)

_LOCAL_ONLY = " (only QLocalServer/QLocalSocket are allowed)"
_PYTHON_SUFFIXES = (".py", ".pyw")
_SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "node_modules"})


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


def _native_lib_stem(text: str) -> str:
    stem = text.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".dll", ".dylib"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.split(".so", 1)[0]


# --------------------------------------------------------------------------- scanner
class _ModuleScanner:
    """Collects violations for one parsed module."""

    def __init__(self, path: Path, tree: ast.Module) -> None:
        self.path = path
        self.tree = tree
        self.violations: list[Violation] = []
        # Names bound to the QtNetwork module in this file ("QtNetwork", "qn", ...).
        self.qt_network_aliases: set[str] = {f"{binding}.QtNetwork" for binding in QT_BINDINGS}

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
        top = module.split(".", 1)[0]
        if top in NETWORK_MODULES:
            self.report(node, "network-import", f"{how} of networking module '{module}'")
            return
        parts = module.split(".")
        if top in QT_BINDINGS and len(parts) >= 2:
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

    # --------------------------------------------------------------- references
    def _check_attribute(self, node: ast.Attribute) -> set[int]:
        """Check one attribute access; returns ids of child nodes to skip."""
        if node.attr == "fourcc" and self._is_video_writer(node.value):
            # cv2.VideoWriter.fourcc("M", "J", "P", "G") only builds a codec code
            # (used to request MJPG from a camera); it writes nothing.
            return {id(node.value)}
        if node.attr in FRAME_WRITE_NAMES:
            self.report(node, "frame-write", f"use of '{node.attr}' (frames must never be encoded)")
        if node.attr.lower() in NATIVE_NETWORK_LIBS:
            self.report(node, "native-network", f"native networking library '{node.attr}'")
        base = _dotted_name(node.value)
        if base in self.qt_network_aliases and node.attr not in QT_NETWORK_ALLOWED:
            self.report(node, "qt-network", f"'{base}.{node.attr}'{_LOCAL_ONLY}")
        return set()

    @staticmethod
    def _is_video_writer(node: ast.AST) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr == "VideoWriter") or (
            isinstance(node, ast.Name) and node.id == "VideoWriter"
        )

    def _check_name(self, node: ast.Name) -> None:
        if node.id in FRAME_WRITE_NAMES:
            self.report(node, "frame-write", f"use of '{node.id}' (frames must never be encoded)")

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
            target = _dotted_name(first)
            if target is not None and target in self.qt_network_aliases:
                attr = _string(node.args[1])
                if attr is None or attr not in QT_NETWORK_ALLOWED:
                    shown = attr if attr is not None else "<dynamic>"
                    self.report(node, "qt-network", f"getattr({target}, {shown!r})")

        if (
            name in _LIBRARY_LOADERS
            and literal is not None
            and _native_lib_stem(literal) in NATIVE_NETWORK_LIBS
        ):
            self.report(node, "native-network", f"loads native networking library {literal!r}")

        if name in _COMMAND_CALLS and literal is not None:
            words = literal.split()
            if words and _command_word(words[0]) in NETWORK_COMMANDS:
                self.report(node, "network-command", f"runs network tool {words[0]!r}")

    def _check_command_sequence(self, node: ast.List | ast.Tuple) -> None:
        # ["curl", url] is flagged wherever it is built, so a command stored in a
        # variable before being passed to subprocess is caught too.
        if not node.elts:
            return
        head = _string(node.elts[0])
        if head is not None and _command_word(head) in NETWORK_COMMANDS:
            self.report(node, "network-command", f"command line runs network tool {head!r}")


# ------------------------------------------------------------------------ public API
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


def _github_annotation(violation: Violation) -> str:
    message = f"{violation.rule}: {violation.message}".replace("%", "%25").replace("\n", "%0A")
    return (
        f"::error file={_display_path(violation.path)},line={violation.line},"
        f"col={violation.col},title=Privacy check::{message}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail if the source tree contains network or frame-writing code.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"files or directories to scan (default: {_display_path(DEFAULT_TARGET)})",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="print violations only")
    args = parser.parse_args(argv)

    targets: list[Path] = args.paths or [DEFAULT_TARGET]
    missing = [p for p in targets if not p.exists()]
    if missing:
        for path in missing:
            print(f"check_privacy: no such file or directory: {path}", file=sys.stderr)
        return 2

    result = scan_paths(targets)
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
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
