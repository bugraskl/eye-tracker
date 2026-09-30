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

The version comes from ``src/eye_tracker/__init__.py`` (parsed, not imported).
Executable names are mirrored in ``entry.py``, which picks the entry point, and
in ``eye_tracker/platform/autostart.py``, which registers the windowed one.
"""

import ast
import glob
import importlib.metadata
import importlib.util
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, copy_metadata

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
    dependencies (pyobjc, python-xlib) appear only where they apply. Packages
    the lock file overrides away (matplotlib, ...) are simply not installed.
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


def prune_unreferenced_openssl(entries: list) -> list:
    """Remove OpenSSL libraries that no other bundled binary imports.

    They are collected for Qt's TLS plugin, which the app never loads (see
    ``_unwanted``), and Python's ``ssl`` module is excluded. Libraries that are
    still imported (``_hashlib`` needs libcrypto) are kept.
    """
    from PyInstaller.depend import bindepend

    def is_openssl(dest: str) -> bool:
        return Path(dest).name.lower().startswith(("libssl", "libcrypto"))

    def imported_names(src: str) -> set:
        try:
            return {Path(name).name.lower() for name, _path in bindepend.get_imports(src)}
        except Exception as exc:  # unreadable binary: keep everything it might need
            log.warning("Cannot read imports of %s: %s", src, exc)
            return {"*"}

    others = [e for e in entries if e[2] in {"BINARY", "EXTENSION"} and not is_openssl(e[0])]
    needed = set().union(*(imported_names(src) for _dest, src, _kind in others))
    candidates = [e for e in entries if is_openssl(e[0])]
    keep: set = set()
    # Iterate: a kept libssl would in turn need libcrypto.
    while True:
        newly = {
            e[0]
            for e in candidates
            if e[0] not in keep and ("*" in needed or Path(e[0]).name.lower() in needed)
        }
        if not newly:
            break
        keep |= newly
        for dest, src, _kind in candidates:
            if dest in newly:
                needed |= imported_names(src)
    removed = [e[0] for e in candidates if e[0] not in keep]
    if removed:
        log.info("Not bundling unreferenced OpenSSL libraries: %s", ", ".join(removed))
    return [e for e in entries if not is_openssl(e[0]) or e[0] in keep]


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


MODEL_FILES = ("face_landmarker.task", "face_detection_yunet_2023mar.onnx")
_missing_models = [f for f in MODEL_FILES if not (PACKAGE / "vision" / "models" / f).is_file()]
if _missing_models:
    raise SystemExit(
        "eye-tracker.spec: missing model files "
        + ", ".join(_missing_models)
        + "; run `uv run python scripts/fetch_models.py` first"
    )

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

if importlib.util.find_spec("mediapipe") is not None:
    # MediaPipe 1.x is a ctypes wrapper: it loads libmediapipe.{dll,so,dylib} from
    # the "mediapipe.tasks.c" package via importlib.resources, which static
    # analysis cannot see.
    binaries += collect_dynamic_libs("mediapipe")
    hiddenimports.append("mediapipe.tasks.c")
else:
    log.warning("MediaPipe is not installed; the bundle will only have the OpenCV backend.")

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
    # MediaPipe's drawing helpers import matplotlib; the app installs a shim instead.
    "matplotlib",
    "sounddevice",
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
for _model in MODEL_FILES:
    if f"eye_tracker/vision/models/{_model}" not in _bundled:
        raise SystemExit(f"eye-tracker.spec: model {_model} was not collected")


def _unwanted(dest: str) -> bool:
    """Qt files that the app never loads.

    * tls / networkinformation: QtNetwork backends for TCP/SSL; only local IPC is used.
    * generic: input plugins for embedded (eglfs) targets, including a UDP (TUIO) listener.
    * opengl32sw.dll: 20 MB software OpenGL fallback; the UI is plain raster widgets.
    * translations: Qt's own UI strings; the app installs no QTranslator.
    """
    path = "/" + dest.replace("\\", "/").lower()
    if path.endswith("/opengl32sw.dll"):
        return True
    if "/pyside6/" in path and "/translations/" in path:
        return True
    return any(f"/plugins/{group}/" in path for group in ("tls", "networkinformation", "generic"))


a.binaries = prune_unreferenced_openssl([e for e in a.binaries if not _unwanted(e[0])])
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
