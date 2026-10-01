# -*- mode: python -*-
"""PyInstaller recipe for Eye Tracker: one-folder builds for Windows, macOS and Linux.

Build from the repository root (the release workflow does exactly this):

    uv run python scripts/fetch_models.py --check   # models present and verified
    uv run python scripts/make_icons.py             # .ico / .icns / .png from ui.icons
    uv run pyinstaller packaging/pyinstaller/eye-tracker.spec --noconfirm --clean

Output in dist/ (or --distpath):

    Windows  EyeTracker/EyeTracker.exe        windowed tray app (no console window)
             EyeTracker/eye-tracker-cli.exe   console CLI (doctor, ctl, bench, autostart ...)
    macOS    Eye Tracker.app                  menu-bar app (LSUIElement, no Dock icon)
             .../Contents/MacOS/eye-tracker-cli
    Linux    eye-tracker/eye-tracker          one executable for the app and the CLI

The version comes from ``src/eye_tracker/__init__.py`` and the list of face
models (with their pinned SHA-256) from ``eye_tracker/vision/backends/__init__.py``;
both are parsed, not imported. Executable names are mirrored in ``entry.py``,
which picks the entry point, and in ``eye_tracker/platform/autostart.py``, which
registers the windowed one.

After the build, the release workflow (and .github/workflows/bundle.yml, on pull
requests that change the packaging or the dependencies) runs the bundle privacy gate:

    uv run python scripts/check_privacy.py --bundle dist/<DIST_NAME>

The Windows installer additionally installs eye-tracker-cli.exe as eye-tracker.exe
(packaging/windows/installer.iss), the name the documentation uses on PATH.
"""

import ast
import collections
import glob
import hashlib
import importlib.metadata
import importlib.util
import logging
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

log = logging.getLogger("eye-tracker.spec")

SPEC_DIR = Path(SPECPATH).resolve()  # noqa: F821 - SPECPATH is injected by PyInstaller
ROOT = SPEC_DIR.parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "eye_tracker"
PACKAGING = ROOT / "packaging"
PROJECT = "eye-tracker"  # distribution name in pyproject.toml

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


# ---------------------------------------------------------------------------- metadata
def read_package_constants() -> dict:
    """String constants (``__version__``, ``APP_NAME``, ...) from the package ``__init__``."""
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            constants[node.targets[0].id] = node.value.value
    return constants


META = read_package_constants()
VERSION = META["__version__"]
APP_NAME = META["APP_NAME"]  # "Eye Tracker"
APP_ID = META["APP_ID"]  # "io.github.bugraskl.eyetracker"
REPO_URL = META["REPO_URL"]
COPYRIGHT = "Copyright © 2026 Buğra Şıkel. MIT License."

_numeric = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", VERSION)
if _numeric is None:
    raise SystemExit(f"eye-tracker.spec: cannot parse version {VERSION!r}")
# Windows resources and CFBundleVersion need a purely numeric version.
VERSION_TUPLE = (*(int(part or 0) for part in _numeric.groups()), 0)
VERSION_NUMERIC = ".".join(str(part) for part in VERSION_TUPLE[:3])

if IS_WINDOWS:
    GUI_NAME, CLI_NAME, DIST_NAME = "EyeTracker", "eye-tracker-cli", "EyeTracker"
elif IS_MACOS:
    GUI_NAME, CLI_NAME, DIST_NAME = APP_NAME, "eye-tracker-cli", APP_NAME
else:
    # Linux has no console/windowed split: one executable serves both roles.
    GUI_NAME, CLI_NAME, DIST_NAME = None, "eye-tracker", "eye-tracker"


def optional_icon(path: Path):
    if path.is_file():
        return str(path)
    log.warning("Icon %s is missing; run scripts/make_icons.py. Using the default icon.", path)
    return None


ICON_WINDOWS = optional_icon(PACKAGING / "windows" / "eye-tracker.ico") if IS_WINDOWS else None
ICON_MACOS = optional_icon(PACKAGING / "macos" / "eye-tracker.icns") if IS_MACOS else None


def runtime_distributions() -> list:
    """Installed distributions the app needs at run time (its dependency closure).

    Environment markers are evaluated for this machine, so platform-only
    dependencies (pyobjc, python-xlib) appear only where they apply.
    """
    found = {}
    pending = [PROJECT]
    while pending:
        name = pending.pop()
        key = canonicalize_name(name)
        if key in found:
            continue
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        found[key] = dist.metadata["Name"]
        for text in dist.requires or ():
            try:
                requirement = Requirement(text)
            except InvalidRequirement:
                continue
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    if canonicalize_name(PROJECT) not in found:
        raise SystemExit(
            "eye-tracker.spec: the project is not installed in this environment; run `uv sync`"
        )
    # The project's own dist-info is not bundled: an editable install records the
    # build machine's source path, and the version is compiled in anyway.
    del found[canonicalize_name(PROJECT)]
    return sorted(found.values(), key=str.lower)


# ------------------------------------------------------------------ what goes in the bundle
def package_modules() -> list:
    """Every module of the package, found on disk.

    Platform integrations and backends are imported lazily (importlib), which
    PyInstaller's static analysis cannot follow, so all of them are listed.
    """
    modules = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        elif parts[-1] == "__main__":
            continue
        modules.append(".".join(parts))
    return modules


def package_data() -> list:
    """Non-Python files of the package (the face models and any future assets)."""
    datas = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix in {".py", ".pyc", ".pyo", ".pyi"}:
            continue
        datas.append((str(path), path.parent.relative_to(SRC).as_posix()))
    return datas


def linux_system_library(soname: str):
    """Absolute path of a system shared library for this architecture, or None."""
    arch_tag = {"x86_64": "x86-64", "aarch64": "AArch64"}.get(platform.machine(), "")
    ldconfig = shutil.which("ldconfig") or next(
        (p for p in ("/sbin/ldconfig", "/usr/sbin/ldconfig") if os.path.exists(p)), None
    )
    if ldconfig:
        try:
            listing = subprocess.run(
                [ldconfig, "-p"], capture_output=True, text=True, check=False, timeout=30
            ).stdout
        except (OSError, subprocess.SubprocessError):
            listing = ""
        # Lines look like: "\tlibxcb-cursor.so.0 (libc6,x86-64) => /lib/x86_64-linux-gnu/..."
        for line in listing.splitlines():
            name, _, rest = line.strip().partition(" (")
            flags, _, path = rest.partition(") => ")
            if name == soname and (not arch_tag or arch_tag in flags) and os.path.isfile(path):
                return path
    for pattern in (f"/usr/lib/*-linux-gnu/{soname}", f"/usr/lib64/{soname}", f"/usr/lib/{soname}"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def hide_foreign_openssl_from_path() -> None:
    """Windows: drop PATH entries that ship OpenSSL DLLs of other software.

    PyInstaller's QtNetwork hook bundles whatever OpenSSL it finds on PATH (Git
    for Windows puts one there), and the first copy found also replaces Python's
    own libcrypto for ``_hashlib``. The bundle must not depend on the build
    machine's PATH, so those folders are hidden while the spec runs.
    """
    python_dirs = {Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()}

    def is_foreign_openssl(entry: str) -> bool:
        if not entry:
            return False
        try:
            folder = Path(entry).resolve()
            if not folder.is_dir() or not any(folder.glob("libssl-*.dll")):
                return False
        except OSError:
            return False
        return not any(d == folder or d in folder.parents for d in python_dirs)

    kept, dropped = [], []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        (dropped if is_foreign_openssl(entry) else kept).append(entry)
    if dropped:
        log.info("Hiding OpenSSL folders from PATH during the build: %s", "; ".join(dropped))
        os.environ["PATH"] = os.pathsep.join(kept)


def _file_name(dest: str) -> str:
    """Lower-case file name of a TOC destination, whichever separator it uses."""
    return dest.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _without(entries: list, dropped: set) -> list:
    """``entries`` minus the BINARY entries whose destination is in ``dropped``.

    Symbolic links that PyInstaller adds on Linux and macOS (``libfoo.so.1`` in
    the top-level folder, pointing at ``pkg.libs/libfoo.so.1``) go with their
    target, so no dangling link is left behind.
    """
    gone = {dest.replace("\\", "/") for dest in dropped}
    return [
        (dest, src, kind)
        for dest, src, kind in entries
        if not (kind == "BINARY" and dest.replace("\\", "/") in gone)
        and not (kind == "SYMLINK" and src.replace("\\", "/") in gone)
    ]


def prune_unreferenced_openssl(entries: list) -> list:
    """Remove OpenSSL libraries that no other bundled binary imports.

    They are collected for Qt's TLS plugin, which the app never loads (see
    ``_unwanted``), and Python's ``ssl`` module is excluded. Libraries that are
    still imported (``_hashlib`` needs libcrypto) are kept.
    """
    from PyInstaller.depend import bindepend

    def is_openssl(dest: str) -> bool:
        return _file_name(dest).startswith(("libssl", "libcrypto"))

    def imported_names(src: str) -> set:
        try:
            return {_file_name(name) for name, _path in bindepend.get_imports(src)}
        except Exception as exc:  # unreadable binary: keep everything it might need
            log.warning("Cannot read imports of %s: %s", src, exc)
            return {"*"}

    others = [e for e in entries if e[2] in {"BINARY", "EXTENSION"} and not is_openssl(e[0])]
    needed = set().union(*(imported_names(src) for _dest, src, _kind in others))
    candidates = [e for e in entries if e[2] == "BINARY" and is_openssl(e[0])]
    keep: set = set()
    # Iterate: a kept libssl would in turn need libcrypto.
    while True:
        newly = {
            e[0]
            for e in candidates
            if e[0] not in keep and ("*" in needed or _file_name(e[0]) in needed)
        }
        if not newly:
            break
        keep |= newly
        for dest, src, _kind in candidates:
            if dest in newly:
                needed |= imported_names(src)
    removed = {e[0] for e in candidates if e[0] not in keep}
    if removed:
        log.info("Not bundling unreferenced OpenSSL libraries: %s", ", ".join(sorted(removed)))
    return _without(entries, removed)


def prune_orphaned_libraries(kept: list, removed: list) -> list:
    """Drop the libraries that only ``removed`` files needed.

    Analysis collects the link-time dependencies of every binary, including the
    Qt plugins that ``_unwanted`` removes afterwards, so removing a plugin leaves
    its dependencies behind. On Linux the GTK3 platform theme alone drags in
    GTK, Pango, Cairo and GIO, whose resolver fails the bundle privacy gate and
    which loads modules and schemas from the host system at run time.

    A kept library is dropped only when a removed file depends on it (directly
    or through other bundled libraries) and no kept file does. Every other kept
    binary counts as needed, including libraries loaded at run time that appear
    in no import table (OpenCV's FFmpeg DLL on Windows). Python extension
    modules are never dropped. If the imports of a needed file cannot be read,
    nothing is dropped.

    The result is the bundle as if the removed files had never been collected:
    the dependency analysis would not have found a library that only they link.
    (A library that a hook collects explicitly would also go if only removed
    files linked it; none of the explicitly collected ones is in that position:
    OpenSSL is loaded at run time and libxcb-cursor is linked by the xcb plugin.)
    """
    from PyInstaller.depend import bindepend

    cache: dict = {}

    def imports(src: str):
        """Lower-case names of the libraries ``src`` links; ``None`` if unreadable."""
        if src not in cache:
            try:
                cache[src] = {_file_name(name) for name, _path in bindepend.get_imports(src)}
            except Exception as exc:
                log.warning("Cannot read imports of %s: %s", src, exc)
                cache[src] = None
        return cache[src]

    libraries: dict = {}  # file name -> kept BINARY/EXTENSION entries of that name
    for entry in kept:
        if entry[2] in {"BINARY", "EXTENSION"}:
            libraries.setdefault(_file_name(entry[0]), []).append(entry)

    def reachable(start: list):
        """Bundled library names reachable from the files ``start``; ``None`` if unknown."""
        found: set = set()
        pending = [src for _dest, src, _kind in start]
        while pending:
            names = imports(pending.pop())
            if names is None:
                return None
            for name in names - found:
                if name in libraries:
                    found.add(name)
                    pending.extend(src for _dest, src, _kind in libraries[name])
        return found

    # What the removed files pulled in. An unreadable removed file adds nothing,
    # which only means that fewer libraries are dropped.
    suspects: set = set()
    for entry in removed:
        if entry[2] in {"BINARY", "EXTENSION"}:
            suspects |= reachable([entry]) or set()
    if not suspects:
        return kept
    roots = [e for name, group in libraries.items() if name not in suspects for e in group]
    needed = reachable(roots)
    if needed is None:
        log.warning("Keeping every dependency of the removed Qt plugins (unreadable imports)")
        return kept
    orphans = {
        name
        for name in suspects - needed
        if all(kind == "BINARY" for _dest, _src, kind in libraries[name])
    }
    if orphans:
        log.info(
            "Not bundling libraries that only removed plugins need: %s", ", ".join(sorted(orphans))
        )
    return _without(kept, {e[0] for name in orphans for e in libraries[name]})


# --------------------------------------------- macOS: libraries that no bundled binary uses
# From <mach-o/loader.h>. A binary's dylib load commands are its library ordinals 1, 2, ...
_MH_MAGIC_64 = 0xFEEDFACF
_MH_TWOLEVEL = 0x80
_LC_SYMTAB = 0x2
_LC_LOAD_DYLIB = 0xC
_LC_LAZY_LOAD_DYLIB = 0x20
_LC_DYLD_INFO = 0x22
_LC_LOAD_WEAK_DYLIB = 0x80000018
_LC_REEXPORT_DYLIB = 0x8000001F
_LC_DYLD_INFO_ONLY = 0x80000022
_LC_LOAD_UPWARD_DYLIB = 0x80000023
_LC_DYLD_CHAINED_FIXUPS = 0x80000034
_DYLIB_LOAD_COMMANDS = (
    _LC_LOAD_DYLIB,
    _LC_LAZY_LOAD_DYLIB,
    _LC_LOAD_WEAK_DYLIB,
    _LC_REEXPORT_DYLIB,
    _LC_LOAD_UPWARD_DYLIB,
)


def _macho_string(data: bytes, offset: int) -> tuple:
    """The NUL-terminated string at ``offset`` and the offset after it."""
    end = data.index(b"\0", offset)  # ValueError when it is not terminated
    return data[offset:end].decode("utf-8", "replace"), end + 1


def _uleb128(data: bytes, position: int) -> tuple:
    """An unsigned LEB128 number at ``position`` and the position after it."""
    value = shift = 0
    while True:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, position


def _bind_opcodes(data: bytes, start: int, size: int) -> list:
    """``(library ordinal, symbol)`` of every bind in a classic dyld-info bind stream."""
    binds = []
    ordinal, symbol = 0, ""
    position, end = start, start + size
    while position < end:
        byte = data[position]
        position += 1
        opcode, immediate = byte & 0xF0, byte & 0x0F
        if opcode == 0x00:  # DONE: lazy bind streams separate their entries with it
            continue
        if opcode == 0x10:  # SET_DYLIB_ORDINAL_IMM
            ordinal = immediate
        elif opcode == 0x20:  # SET_DYLIB_ORDINAL_ULEB
            ordinal, position = _uleb128(data, position)
        elif opcode == 0x30:  # SET_DYLIB_SPECIAL_IMM: 0 (self), -1, -2 or -3
            ordinal = immediate - 16 if immediate else 0
        elif opcode == 0x40:  # SET_SYMBOL_TRAILING_FLAGS_IMM
            symbol, position = _macho_string(data, position)
        elif opcode == 0x50:  # SET_TYPE_IMM
            pass
        elif opcode in (0x60, 0x70, 0x80):  # addend (SLEB), segment offset, address step
            _value, position = _uleb128(data, position)  # an SLEB128 is skipped the same way
        elif opcode in (0x90, 0xB0):  # DO_BIND, DO_BIND_ADD_ADDR_IMM_SCALED
            binds.append((ordinal, symbol))
        elif opcode == 0xA0:  # DO_BIND_ADD_ADDR_ULEB
            _value, position = _uleb128(data, position)
            binds.append((ordinal, symbol))
        elif opcode == 0xC0:  # DO_BIND_ULEB_TIMES_SKIPPING_ULEB
            _value, position = _uleb128(data, position)
            _value, position = _uleb128(data, position)
            binds.append((ordinal, symbol))
        elif byte == 0xD0:  # THREADED: SET_BIND_ORDINAL_TABLE_SIZE_ULEB
            _value, position = _uleb128(data, position)
        elif byte != 0xD1:  # THREADED: APPLY
            raise ValueError(f"unknown bind opcode {byte:#04x}")
    return binds


def _macho_slice(data: bytes, base: int, info: dict) -> None:
    """Add what the 64-bit Mach-O image at ``base`` links, binds, looks up and defines."""
    magic, _cpu, _subtype, _filetype, count, _size, flags = struct.unpack_from("<7I", data, base)
    if magic != _MH_MAGIC_64:
        raise ValueError("not a 64-bit little-endian Mach-O image")
    names = []  # install names, by library ordinal - 1
    symtab = fixups = dyld_info = None
    position = base + 32
    for _ in range(count):
        command, size = struct.unpack_from("<II", data, position)
        if size < 8:
            raise ValueError("corrupt Mach-O load command")
        if command in _DYLIB_LOAD_COMMANDS:
            (name_offset,) = struct.unpack_from("<I", data, position + 8)
            name, _end = _macho_string(data, position + name_offset)
            names.append(name)
            info["links"].append((position, command, name))
        elif command == _LC_SYMTAB:
            symtab = struct.unpack_from("<4I", data, position + 8)
        elif command == _LC_DYLD_CHAINED_FIXUPS:
            fixups = struct.unpack_from("<2I", data, position + 8)
        elif command in (_LC_DYLD_INFO, _LC_DYLD_INFO_ONLY):
            dyld_info = struct.unpack_from("<8I", data, position + 8)
        position += size

    twolevel = bool(flags & _MH_TWOLEVEL)
    if not twolevel:
        # Flat namespace: every symbol is looked up in every image, so any
        # linked library may be the one that provides it.
        info["bound"].update(names)

    def bind(ordinal: int, symbol: str) -> None:
        if not twolevel or ordinal in (-2, -3):  # flat namespace, flat or weak lookup
            info["lookups"].add(symbol)
        elif 0 < ordinal <= len(names):
            info["bound"].add(names[ordinal - 1])
        elif ordinal not in (0, -1):  # 0 is this image, -1 the main executable
            raise ValueError(f"unknown library ordinal {ordinal}")

    if symtab:  # what `nm -m` shows
        symbols, symbol_count, strings, _strings_size = symtab
        for index in range(symbol_count):
            name_at, kind, _section, desc, _value = struct.unpack_from(
                "<IBBHQ", data, base + symbols + 16 * index
            )
            if kind & 0xE0 or not kind & 0x01 or not name_at:  # debugging entry, not external
                continue
            symbol, _end = _macho_string(data, base + strings + name_at)
            if (kind & 0x0E) in (0x02, 0x0A, 0x0E):  # N_ABS, N_INDR, N_SECT: defined here
                info["exports"].add(symbol)
            elif (kind & 0x0E) in (0x00, 0x0C):  # N_UNDF, N_PBUD: imported
                ordinal = desc >> 8  # 0xFE: dynamic lookup, 0xFF: the main executable
                bind(ordinal - 0x100 if ordinal >= 0xFE else ordinal, symbol)
    if fixups:  # what dyld binds in images with chained fixups
        start = base + fixups[0]
        header = struct.unpack_from("<7I", data, start)
        _version, _starts, imports, symbol_names, import_count, import_format, compressed = header
        if compressed:
            raise ValueError("compressed chained-fixup symbol names")
        # DYLD_CHAINED_IMPORT, _ADDEND (8-bit ordinals) and _ADDEND64 (16-bit ordinals).
        stride = {1: 4, 2: 8, 3: 16}.get(import_format)
        if stride is None:
            raise ValueError(f"unknown chained-fixup import format {import_format}")
        for index in range(import_count):
            if import_format == 3:
                (value,) = struct.unpack_from("<Q", data, start + imports + stride * index)
                ordinal, name_at = value & 0xFFFF, value >> 32
                ordinal -= 0x10000 if ordinal >= 0xFFF0 else 0
            else:
                (value,) = struct.unpack_from("<I", data, start + imports + stride * index)
                ordinal, name_at = value & 0xFF, value >> 9
                ordinal -= 0x100 if ordinal >= 0xF0 else 0
            symbol, _end = _macho_string(data, start + symbol_names + name_at)
            bind(ordinal, symbol)
    if dyld_info:  # what dyld binds in images with classic bind opcodes
        _rebase, _rebase_size, binds, binds_size, weak, weak_size, lazy, lazy_size = dyld_info
        for offset, size in ((binds, binds_size), (lazy, lazy_size)):
            for ordinal, symbol in _bind_opcodes(data, base + offset, size):
                bind(ordinal, symbol)
        # Weak binds coalesce C++ weak definitions across all loaded images.
        for _ordinal, symbol in _bind_opcodes(data, base + weak, weak_size):
            info["lookups"].add(symbol)


def macho_bindings(data: bytes):
    """What a Mach-O file links, and what it binds to those libraries; ``None`` for other files.

    The result has:

    ``links``
        ``(file offset, command, install name)`` of every dylib load command.
    ``bound``
        The install names that at least one symbol is bound to.
    ``lookups``
        Symbols that dyld finds by searching every loaded image (flat namespace,
        ``-undefined dynamic_lookup`` as in Python extension modules, C++ weak
        definitions); any library that defines one of them may be needed.
    ``exports``
        The external symbols the file defines.

    The bindings come from the symbol table (what ``nm -m`` shows), the chained
    fixups and the classic bind opcodes together, every slice of a universal
    file included. Raises ``ValueError`` for a Mach-O file that cannot be read
    completely (32-bit, big-endian, truncated or unknown formats).
    """
    info = {"links": [], "bound": set(), "lookups": set(), "exports": set()}
    magic = data[:4]
    thin = (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf")
    try:
        if magic in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
            (count,) = struct.unpack_from(">I", data, 4)
            if not 0 < count < 20:
                return None  # a Java class file has its version number there
            wide = magic == b"\xca\xfe\xba\xbf"
            for index in range(count):
                if wide:
                    (offset,) = struct.unpack_from(">Q", data, 8 + 32 * index + 8)
                else:
                    (offset,) = struct.unpack_from(">I", data, 8 + 20 * index + 8)
                _macho_slice(data, offset, info)
        elif magic in thin:  # only 64-bit little-endian images can be read (_macho_slice)
            _macho_slice(data, 0, info)
        else:
            return None
    except (struct.error, IndexError) as exc:
        raise ValueError(f"truncated Mach-O file: {exc}") from exc
    return info


def prune_unused_macos_libraries(entries: list, scratch: Path) -> list:
    """macOS: leave out the libraries that no bundled binary binds a symbol to.

    OpenCV's macOS wheel ships FFmpeg as Homebrew builds it, and FFmpeg links
    each of its libraries against the libraries of all its components: libX11
    is linked by eight FFmpeg libraries and used by none, and libsrt links
    libssl without using it. dyld loads every linked library and refuses to
    load a binary when one is missing, so such a link is made weak in a copy of
    the linking binary (only the load command's type changes; library ordinals
    stay as they are) and the library is not bundled, which is what
    ``ld -dead_strip_dylibs`` would have done when the libraries were built.

    A library goes only when at least one bundled binary links it (libraries
    loaded with dlopen(), such as Qt plugins, are linked by nothing and stay),
    and every bundled binary that still links it does so with a plain or weak
    load command (not a re-export or upward link) and binds no symbol to it.
    It also stays when any remaining binary looks up by name a symbol it
    defines (see ``macho_bindings``). Python extension modules always stay. A
    library that only removed libraries linked goes too. If any Mach-O file
    cannot be read, nothing is removed.

    The PyInstaller build then rewrites and re-signs the copies like any other
    collected binary. Returns the new TOC list.
    """
    infos = {}
    for dest, src, kind in entries:
        if kind not in {"BINARY", "EXTENSION"}:
            continue
        try:
            info = macho_bindings(Path(src).read_bytes())
        except (OSError, ValueError) as exc:
            log.warning(
                "Keeping every linked library; cannot read the bindings of %s: %s", src, exc
            )
            return entries
        if info is not None:
            infos[dest] = (kind, info)

    by_name: dict = {}
    for dest in infos:
        by_name.setdefault(_file_name(dest), []).append(dest)

    def targets(name: str) -> list:
        return by_name.get(_file_name(name), [])

    users: dict = {}  # bundled library -> the bundled binaries that link it
    for dest, (_kind, info) in infos.items():
        for _offset, _command, name in info["links"]:
            for target in targets(name):
                if target != dest:
                    users.setdefault(target, set()).add(dest)

    def binds_nothing(user: str, target: str) -> bool:
        info = infos[user][1]
        return all(
            command in (_LC_LOAD_DYLIB, _LC_LOAD_WEAK_DYLIB) and name not in info["bound"]
            for _offset, command, name in info["links"]
            if target in targets(name)
        )

    unused: set = set()
    while True:
        lookups = collections.Counter(
            symbol
            for dest, (_kind, info) in infos.items()
            if dest not in unused
            for symbol in info["lookups"]
        )
        newly = set()
        for target, linked_by in users.items():
            kind, info = infos[target]
            if target in unused or kind != "BINARY":
                continue
            if any(lookups[symbol] > (symbol in info["lookups"]) for symbol in info["exports"]):
                continue  # another binary may find one of its symbols by name
            if all(binds_nothing(user, target) for user in linked_by - unused):
                newly.add(target)
        if not newly:
            break
        unused |= newly
    if not unused:
        return entries

    weakened = {}
    for dest, (_kind, info) in infos.items():
        if dest in unused:
            continue
        offsets = [
            offset
            for offset, command, name in info["links"]
            if command == _LC_LOAD_DYLIB and any(target in unused for target in targets(name))
        ]
        if offsets:
            weakened[dest] = offsets
    result = []
    for dest, src, kind in entries:
        if dest in weakened:
            data = bytearray(Path(src).read_bytes())
            for offset in weakened[dest]:
                struct.pack_into("<I", data, offset, _LC_LOAD_WEAK_DYLIB)
            copy = scratch / dest
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(data)
            src = str(copy)
        result.append((dest, src, kind))
    log.info(
        "Not bundling libraries that no bundled binary uses: %s (now weakly linked by %s)",
        ", ".join(sorted(_file_name(dest) for dest in unused)),
        ", ".join(sorted(_file_name(dest) for dest in weakened)),
    )
    return _without(result, unused)


def macos_minimum_version(distributions, floor=(12, 0)) -> str:
    """The oldest macOS every bundled wheel supports (never below ``floor``).

    A wheel tagged ``macosx_13_0_*`` refuses to load on older systems, so the
    bundle's LSMinimumSystemVersion is the newest such tag across the bundle.
    """
    best = tuple(floor)
    for name in distributions:
        try:
            wheel = importlib.metadata.distribution(name).read_text("WHEEL") or ""
        except importlib.metadata.PackageNotFoundError:
            continue
        tags = re.findall(r"^Tag:\s*\S+-macosx_(\d+)_(\d+)_\w+\s*$", wheel, re.MULTILINE)
        if tags:
            best = max(best, min((int(major), int(minor)) for major, minor in tags))
    return f"{best[0]}.{best[1]}"


def pinned_models() -> dict:
    """``MODEL_FILES`` (file name -> SHA-256) from ``eye_tracker.vision.backends``.

    Parsed rather than imported, so the spec does not load OpenCV; the backends
    and ``scripts/fetch_models.py`` pin the same files.
    """
    source = (PACKAGE / "vision" / "backends" / "__init__.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == "MODEL_FILES" and value is not None:
            return dict(ast.literal_eval(value))
    raise SystemExit("eye-tracker.spec: MODEL_FILES not found in eye_tracker/vision/backends")


def verify_models(models: dict, directory: Path) -> None:
    """Refuse to build with a missing or modified model file."""
    problems = []
    for name, expected in sorted(models.items()):
        path = directory / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"{name} does not match its pinned SHA-256")
    if problems:
        raise SystemExit(
            "eye-tracker.spec: "
            + "; ".join(problems)
            + ". Run `uv run python scripts/fetch_models.py` first."
        )


def pinned_licences() -> dict:
    """``LICENCE_TEXTS`` (path below the models -> SHA-256) from ``scripts/fetch_models.py``.

    The models' licence texts must ship with them: Apache-2.0 asks for a copy of
    the licence, MIT for its copyright and permission notice. Parsed like
    :func:`pinned_models`, so the spec and ``fetch_models.py --check`` pin the
    same files.
    """
    source = (ROOT / "scripts" / "fetch_models.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and target.id == "LICENCE_TEXTS"):
            continue
        licences = {}
        for call in value.elts if isinstance(value, ast.Tuple) else ():
            fields = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in getattr(call, "keywords", ())
                if keyword.arg in ("path", "sha256")
            }
            if len(fields) == 2:
                licences[fields["path"]] = fields["sha256"]
        if licences:
            return licences
    raise SystemExit("eye-tracker.spec: LICENCE_TEXTS not found in scripts/fetch_models.py")


def verify_licences(licences: dict, directory: Path) -> None:
    """Refuse to build without the models' licence texts, or with altered ones."""
    problems = []
    for name, expected in sorted(licences.items()):
        path = directory / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"{name} does not match its pinned SHA-256")
    if problems:
        # Committed, never downloaded: only git has the right text.
        raise SystemExit(
            "eye-tracker.spec: " + "; ".join(problems) + ". Restore the licence texts from git."
        )


MODELS_DIR = PACKAGE / "vision" / "models"
MODEL_FILES = pinned_models()
verify_models(MODEL_FILES, MODELS_DIR)
# Shipped with the models: their licences and provenance (Apache-2.0 requires it).
MODEL_NOTICE = "NOTICE.md"
if not (MODELS_DIR / MODEL_NOTICE).is_file():
    raise SystemExit(f"eye-tracker.spec: {MODELS_DIR / MODEL_NOTICE} is missing")
MODEL_LICENCES = pinned_licences()
verify_licences(MODEL_LICENCES, MODELS_DIR)

if IS_WINDOWS:
    hide_foreign_openssl_from_path()

RUNTIME_DISTRIBUTIONS = runtime_distributions()
log.info("Runtime distributions: %s", ", ".join(RUNTIME_DISTRIBUTIONS))

hiddenimports = package_modules()
datas = package_data()
binaries = []

# dist-info folders let importlib.metadata report library versions ("doctor").
for _dist in RUNTIME_DISTRIBUTIONS:
    datas += copy_metadata(_dist)

if IS_MACOS:
    # platform/macos.py imports pyobjc lazily with importlib.import_module(), and
    # the framework wrappers load parts of themselves dynamically.
    for _name in (
        "objc",
        "Foundation",
        "AppKit",
        "CoreFoundation",
        "Quartz",
        "ApplicationServices",
        "AVFoundation",
    ):
        if importlib.util.find_spec(_name) is not None:
            hiddenimports += collect_submodules(_name)
        else:
            log.warning("pyobjc module %s is not installed; macOS features will be missing", _name)

if IS_LINUX:
    # Qt's xcb platform plugin (Qt >= 6.5) needs libxcb-cursor.so.0, which many
    # desktops (e.g. Ubuntu 22.04) do not install by default. Bundle it next to
    # the other libraries; the PyInstaller bootloader puts that folder on
    # LD_LIBRARY_PATH.
    _xcb_cursor = linux_system_library("libxcb-cursor.so.0")
    if _xcb_cursor:
        binaries.append((_xcb_cursor, "."))
    else:
        log.warning(
            "libxcb-cursor.so.0 not found (apt install libxcb-cursor0); the bundle will need "
            "it on the target system for the X11 (xcb) platform plugin."
        )

EXCLUDES = [
    # The MediaPipe *runtime* must never ship: it contains a usage logger that
    # uploads to Google. Its face models run in OpenCV DNN instead (see
    # vision/models/NOTICE.md); the bundle privacy gate fails if it slips in.
    "mediapipe",
    # No TLS: the app never uses the network (scripts/check_privacy.py enforces it
    # for our code; this keeps a TLS stack out of the bundle as well).
    "ssl",
    "_ssl",
    # Build and dev tooling that must never end up in the bundle.
    "PIL",
    "PyInstaller",
    "pip",
    "pkg_resources",
    "setuptools",
    "pytest",
    "_pytest",
    "mypy",
    "IPython",
    "tkinter",
    "_tkinter",
    # Qt modules the app does not use (several pull in QML, Chromium or multimedia stacks).
    *(
        f"PySide6.{module}"
        for module in (
            "Qt3DAnimation", "Qt3DCore", "Qt3DExtras", "Qt3DInput", "Qt3DLogic", "Qt3DRender",
            "QtAxContainer", "QtBluetooth", "QtCharts", "QtConcurrent", "QtDataVisualization",
            "QtDesigner", "QtGraphs", "QtGraphsWidgets", "QtHelp", "QtHttpServer", "QtLocation",
            "QtMultimedia", "QtMultimediaWidgets", "QtNetworkAuth", "QtNfc", "QtPdf",
            "QtPdfWidgets", "QtPositioning", "QtPrintSupport", "QtQml", "QtQuick", "QtQuick3D",
            "QtQuickControls2", "QtQuickTest", "QtQuickWidgets", "QtRemoteObjects", "QtScxml",
            "QtSensors", "QtSerialBus", "QtSerialPort", "QtSpatialAudio", "QtSql",
            "QtStateMachine", "QtTest", "QtTextToSpeech", "QtUiTools", "QtWebChannel",
            "QtWebEngineCore", "QtWebEngineQuick", "QtWebEngineWidgets", "QtWebSockets",
            "QtWebView", "QtXml",
        )
    ),
]

a = Analysis(  # noqa: F821 - injected by PyInstaller
    [str(SPEC_DIR / "entry.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

_bundled = {Path(dest).as_posix() for dest, _src, _kind in a.datas}
for _model in (*MODEL_FILES, MODEL_NOTICE, *MODEL_LICENCES):
    if f"eye_tracker/vision/models/{_model}" not in _bundled:
        raise SystemExit(f"eye-tracker.spec: {_model} was not collected")


def _unwanted(dest: str) -> bool:
    """Qt files that the app never loads (and that must not ship).

    * tls, networkinformation, networkaccess: QtNetwork's TLS, reachability and
      HTTP backends; the app only uses local IPC (QLocalServer/QLocalSocket).
    * generic: input plugins for embedded (eglfs) targets, including a UDP (TUIO) listener.
    * platforms vnc: a VNC server (TCP port 5900, no authentication) that an
      inherited QT_QPA_PLATFORM=vnc would start; webgl (Qt 5) is an HTTP server.
    * platforms eglfs, linuxfb, minimalegl, vkkhrdisplay and egldeviceintegrations:
      full-screen targets without a desktop, where a tray app cannot run; they
      link libinput, libudev and libgbm from the build machine.
    * platformthemes gtk3: links GTK, Pango, Cairo and GIO (a DNS resolver) from
      the build machine, whose GIO modules and schemas then come from the host and
      break on other distributions. Qt's built-in GNOME/KDE themes and the
      xdg-desktop-portal theme remain.
    * opengl32sw.dll: 20 MB software OpenGL fallback; the UI is plain raster widgets.
    * translations: Qt's own UI strings; the app installs no QTranslator.

    ``prune_orphaned_libraries`` then removes what only these files linked.
    """
    path = "/" + dest.replace("\\", "/").lower()
    name = path.rsplit("/", 1)[-1]
    if name == "opengl32sw.dll":
        return True
    if "/pyside6/" in path and "/translations/" in path:
        return True
    unwanted_groups = (
        "tls",
        "networkinformation",
        "networkaccess",
        "generic",
        "egldeviceintegrations",
    )
    if any(f"/plugins/{group}/" in path for group in unwanted_groups):
        return True
    if "/plugins/platforms/" in path:
        return any(
            kind in name
            for kind in ("vnc", "webgl", "eglfs", "linuxfb", "minimalegl", "vkkhrdisplay")
        )
    if "/plugins/platformthemes/" in path:
        return "gtk" in name
    return False


_removed_binaries = [entry for entry in a.binaries if _unwanted(entry[0])]
a.binaries = prune_unreferenced_openssl(
    prune_orphaned_libraries(
        [entry for entry in a.binaries if not _unwanted(entry[0])], _removed_binaries
    )
)
if IS_MACOS:
    # libX11, libssl and libhwy, which OpenCV's wheel links but never uses (see the function).
    a.binaries = prune_unused_macos_libraries(
        a.binaries, Path(workpath) / "unused-libraries"  # noqa: F821 - injected by PyInstaller
    )
a.datas = [entry for entry in a.datas if not _unwanted(entry[0])]

pyz = PYZ(a.pure)  # noqa: F821 - injected by PyInstaller


# ------------------------------------------------------------------------ executables
def windows_version_info(original_filename: str, description: str):
    if not IS_WINDOWS:
        return None
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo,
        StringFileInfo,
        StringStruct,
        StringTable,
        VarFileInfo,
        VarStruct,
        VSVersionInfo,
    )

    strings = [
        StringStruct("CompanyName", META.get("APP_AUTHOR", "bugraskl")),
        StringStruct("FileDescription", description),
        StringStruct("FileVersion", VERSION),
        StringStruct("InternalName", Path(original_filename).stem),
        StringStruct("LegalCopyright", COPYRIGHT),
        StringStruct("OriginalFilename", original_filename),
        StringStruct("ProductName", APP_NAME),
        StringStruct("ProductVersion", VERSION),
        StringStruct("Comments", REPO_URL),
    ]
    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=VERSION_TUPLE, prodvers=VERSION_TUPLE),
        kids=[
            StringFileInfo([StringTable("040904B0", strings)]),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
        ],
    )


def make_exe(name: str, *, console: bool, description: str):
    return EXE(  # noqa: F821 - injected by PyInstaller
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=name,
        console=console,
        # UPX-compressed Qt libraries break and trigger antivirus false positives.
        upx=False,
        strip=False,
        debug=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        icon=ICON_WINDOWS,  # macOS takes its icon from BUNDLE
        version=windows_version_info(f"{name}.exe", description),
        contents_directory="_internal",
    )


cli_exe = make_exe(CLI_NAME, console=True, description=f"{APP_NAME} command-line interface")
executables = [cli_exe]
if GUI_NAME is not None:
    gui_exe = make_exe(GUI_NAME, console=False, description=APP_NAME)
    # COLLECT inherits "console" from the last EXE; the windowed one decides
    # how the macOS bundle is flagged.
    executables.append(gui_exe)

coll = COLLECT(  # noqa: F821 - injected by PyInstaller
    *executables,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=DIST_NAME,
)

if IS_MACOS:
    MACOS_MINIMUM = macos_minimum_version(RUNTIME_DISTRIBUTIONS)
    log.info("LSMinimumSystemVersion: %s (from the bundled wheels)", MACOS_MINIMUM)
    app = BUNDLE(  # noqa: F821 - injected by PyInstaller
        coll,
        name=f"{APP_NAME}.app",
        icon=ICON_MACOS,
        bundle_identifier=APP_ID,
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleExecutable": GUI_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION_NUMERIC,
            "LSMinimumSystemVersion": MACOS_MINIMUM,
            "LSApplicationCategoryType": "public.app-category.productivity",
            # Menu-bar (tray) app: no Dock icon, but it does show UI.
            "LSUIElement": True,
            "LSBackgroundOnly": False,
            "NSHighResolutionCapable": True,
            "NSSupportsAutomaticGraphicsSwitching": True,
            "NSCameraUsageDescription": (
                "Eye Tracker uses the camera to see which monitor you are looking at and whether "
                "you are at your desk. Frames are analysed on this Mac and are never saved or sent "
                "anywhere."
            ),
            "NSHumanReadableCopyright": COPYRIGHT,
        },
    )
