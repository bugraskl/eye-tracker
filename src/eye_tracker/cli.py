"""Command-line entry points: ``eye-tracker`` (console) and ``eye-tracker-gui`` (windowed).

Without a subcommand the tray app starts (``run``). Other subcommands help with
setup and troubleshooting and never need a display, except ``doctor``, which
reports monitors when one is available.

Exit codes: 0 success, 1 error, 2 usage error, 3 the app is not running
(``ctl``), 130 interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import APP_NAME, APP_SLUG, __version__, paths
from .config import Settings
from .logging_setup import LEVELS, setup_logging

__all__ = [
    "EXIT_ERROR",
    "EXIT_NOT_RUNNING",
    "EXIT_OK",
    "EXIT_USAGE",
    "app_command",
    "build_parser",
    "cli_command",
    "cli_command_text",
    "format_command",
    "gui_main",
    "main",
]

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NOT_RUNNING = 3
EXIT_INTERRUPTED = 130

#: Name of the console executable next to the windowed one in the Windows and
#: macOS builds (packaging/pyinstaller/eye-tracker.spec). The Linux build has
#: one executable for both roles.
FROZEN_CLI_NAME = "eye-tracker-cli"

BACKEND_CHOICES = ("auto", "facemesh", "lite")
#: Backend names of Eye Tracker 0.1, still accepted (e.g. in desktop shortcuts).
_LEGACY_BACKENDS = {"mediapipe": "facemesh", "opencv": "lite"}
# Mirrors ipc.COMMANDS; kept here so building the parser does not import Qt.
CTL_COMMANDS = (
    "calibrate",
    "pause",
    "privacy-off",
    "privacy-on",
    "privacy-toggle",
    "quit",
    "resume",
    "settings",
    "show",
    "status",
    "toggle",
)

#: ``--help`` examples: (arguments, explanation).
_EXAMPLES = (
    ("", "start the tray app"),
    ("calibrate", "calibrate (in the running app, if there is one)"),
    ("ctl privacy-toggle", "bind this to a desktop shortcut (e.g. on Wayland)"),
    ("doctor", "show a diagnostics report for bug reports"),
    ("bench --seconds 10", "measure CPU use and latency on this machine"),
    ("autostart enable", "start at login"),
)


def _epilog(prog: str) -> str:
    """Examples spelled with the name this copy is run by (see :func:`cli_command`:
    ``eye-tracker`` exists in source installs and, through the PATH, after the
    Windows installer; the other packages have ``eye-tracker-cli`` or the
    ``.AppImage`` file)."""
    commands = [f"{prog} {args}".rstrip() for args, _ in _EXAMPLES]
    width = max(len(command) for command in commands) + 2
    lines = [
        f"  {command:<{width}}{explanation}"
        for command, (_, explanation) in zip(commands, _EXAMPLES, strict=True)
    ]
    return (
        "examples:\n"
        + "\n".join(lines)
        + "\n\nexit codes: 0 ok, 1 error, 2 usage error, 3 not running (ctl), 130 interrupted\n"
    )


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """The argument parser (global options work before or after the subcommand)."""
    prog = _program_name()
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            f"{APP_NAME}: look at a monitor and the mouse cursor and keyboard focus follow. "
            "Webcam-based, private and offline."
        ),
        epilog=_epilog(prog),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    _add_global_options(parser, suppress=False)
    _add_run_options(parser, suppress=False)

    # Subparsers repeat the global options with SUPPRESS defaults, so a value given
    # after the subcommand wins and an absent one does not reset the main parser's.
    common = argparse.ArgumentParser(add_help=False)
    _add_global_options(common, suppress=True)

    sub = parser.add_subparsers(dest="command", metavar="COMMAND", title="commands")

    run = sub.add_parser(
        "run",
        parents=[common],
        help="start the tray app (the default)",
        description=(
            "Start the tray app. If it is already running, the running instance is shown "
            "instead (with --background, nothing is), and --trace, --camera and --backend are "
            "not applied: they only take effect when the app starts."
        ),
    )
    _add_run_options(run, suppress=True)

    sub.add_parser(
        "calibrate",
        parents=[common],
        help="calibrate now",
        description="Open the calibration in the running app, or start the app and calibrate.",
    )

    doctor = sub.add_parser(
        "doctor",
        parents=[common],
        help="print a diagnostics report",
        description="Print everything useful for a bug report (paths are shown relative to ~).",
    )
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    doctor.add_argument(
        "--probe-cameras",
        action="store_true",
        help="briefly open cameras 0-3 to list the working ones (their lights may flash)",
    )

    bench = sub.add_parser(
        "bench",
        parents=[common],
        help="measure CPU use and latency",
        description=(
            "Run the capture and analysis loop without the UI, first as fast as possible and "
            "then at the idle rate the app uses while you sit still. Use --camera to benchmark "
            "a video or image file instead of the configured camera."
        ),
    )
    bench.add_argument(
        "--seconds",
        type=_positive_float,
        default=5.0,
        metavar="N",
        help="duration of each of the two runs (default: 5)",
    )
    bench.add_argument("--json", action="store_true", help="machine-readable output")

    ctl = sub.add_parser(
        "ctl",
        parents=[common],
        help="send a command to the running app",
        description=(
            "Send a command to the running app. Bind these to desktop keyboard shortcuts where "
            "global hotkeys are unavailable (Wayland). 'status' prints JSON."
        ),
    )
    ctl.add_argument(
        "action", choices=CTL_COMMANDS, metavar="COMMAND", help=", ".join(CTL_COMMANDS)
    )
    ctl.add_argument(
        "--timeout",
        type=int,
        default=1500,
        metavar="MS",
        help="how long to wait for an answer (default: 1500)",
    )

    autostart = sub.add_parser(
        "autostart",
        parents=[common],
        help="start at login: enable, disable or status",
        description="Manage starting the app at login (per user, no administrator rights).",
    )
    autostart.add_argument(
        "action", nargs="?", default="status", choices=("enable", "disable", "status")
    )

    reset = sub.add_parser(
        "reset",
        parents=[common],
        help="delete the calibration and/or settings",
        description="Delete stored data. The app must not be running.",
    )
    reset.add_argument("--calibration", action="store_true", help="delete the calibration")
    reset.add_argument("--settings", action="store_true", help="restore default settings")
    reset.add_argument("--all", action="store_true", help="both of the above")
    return parser


def _add_global_options(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    def default(value: object) -> object:
        return argparse.SUPPRESS if suppress else value

    group = parser.add_argument_group("global options")
    group.add_argument(
        "--config-dir",
        metavar="DIR",
        default=default(None),
        help="keep settings, calibration and logs in DIR (portable mode; a separate instance)",
    )
    group.add_argument(
        "--log-level",
        type=str.upper,
        choices=LEVELS,
        default=default(None),
        help="logging verbosity (default: from the settings, INFO)",
    )
    group.add_argument(
        "--camera",
        metavar="DEV",
        default=default(None),
        help="camera index (0, 1, ...) or a video/image file; overrides the settings",
    )
    group.add_argument(
        "--backend",
        type=_backend_name,
        choices=BACKEND_CHOICES,
        default=default(None),
        help="vision backend: facemesh (head and eyes), lite (head only, lightest) or "
        "auto; overrides the settings",
    )
    group.add_argument(
        "--background",
        action="store_true",
        default=default(False),
        help="start quietly, as at login: no first-run wizard, no startup notifications",
    )


def _add_run_options(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help="open the calibration right after start",
    )
    parser.add_argument(
        "--trace",
        metavar="FILE",
        default=argparse.SUPPRESS if suppress else None,
        help="append numeric tracking data (features, head angles, gaze, pointer position, "
        "decisions; never images) to FILE as JSON lines, for tuning and bug reports; applies "
        "only when the app starts",
    )


def _backend_name(text: str) -> str:
    """A ``--backend`` value, with the 0.1 names mapped to the current ones."""
    name = text.strip().lower()
    return _LEGACY_BACKENDS.get(name, name)


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value > 0 or value == float("inf"):
        raise argparse.ArgumentTypeError("must be a positive number")
    return value


# --------------------------------------------------------- commands shown to users
def cli_command(*args: str) -> list[str]:
    """The command line that runs this installation's command-line interface.

    Hints such as "bind ``… ctl toggle`` to a desktop shortcut" or "run ``…
    doctor``" must name a command that exists here. Only source installs and
    the Windows installer (its ``eye-tracker.exe``, for terminals opened after
    the installation, unless its PATH option was unticked) put an
    ``eye-tracker`` command on the PATH. The result, followed by ``args``, is

    * inside an AppImage: the ``.AppImage`` file (AppRun passes the arguments
      on; the mounted bundle path changes on every run);
    * the Windows installer's copy, when this process's PATH finds its
      ``eye-tracker.exe``: ``eye-tracker``;
    * other frozen builds: the console executable of this copy
      (``eye-tracker-cli.exe`` next to ``EyeTracker.exe``, ``eye-tracker-cli``
      inside the macOS app, the one executable of the Linux bundle);
    * source and pip installs: ``eye-tracker`` when that command on the PATH is
      this installation's, else its script's full path, else
      ``python -m eye_tracker`` with this interpreter.

    A ``--config-dir`` profile is passed on (``--config-dir DIR`` follows the
    program): each profile is a separate instance with its own command socket,
    so ``ctl`` without it would talk to the default profile.
    """
    return [*_cli_base(), *_profile_args(), *args]


def cli_command_text(*args: str) -> str:
    """:func:`cli_command` as the user would type it (quoted for this OS's shell)."""
    return format_command(cli_command(*args))


def app_command() -> list[str]:
    """The command line that starts the tray app of this installation.

    Like :func:`cli_command`, except that the Windows and macOS builds name
    their windowed executable: a console one (``eye-tracker-cli``, or the
    installer's ``eye-tracker.exe``) would tie the app to the terminal it was
    started from.
    """
    base = _cli_base()
    if paths.is_frozen() and _appimage() is None:
        exe = Path(sys.executable)
        for name in _FROZEN_GUI_NAMES:
            windowed = exe.with_name(name + exe.suffix)
            if windowed.is_file():
                base = [str(windowed)]
                break
    return [*base, *_profile_args()]


def format_command(parts: Sequence[str]) -> str:
    """Render a command as the user would type it on this OS."""
    if os.name == "nt":
        return subprocess.list2cmdline(list(parts))
    return shlex.join(parts)


#: Windowed executables next to :data:`FROZEN_CLI_NAME` (Windows, macOS).
_FROZEN_GUI_NAMES = ("EyeTracker", APP_NAME)


def _profile_args() -> list[str]:
    override = paths.base_override()
    return ["--config-dir", str(override)] if override is not None else []


def _cli_base() -> list[str]:
    """The program part of :func:`cli_command` (no profile, no arguments)."""
    if paths.is_frozen():
        appimage = _appimage()
        if appimage is not None:
            return [appimage]
        exe = Path(sys.executable)
        if sys.platform == "win32" and _installer_command_on_path(exe):
            return [APP_SLUG]
        if exe.stem.lower() != FROZEN_CLI_NAME:
            # The windowed executable has no console: its output would be lost.
            console = exe.with_name(FROZEN_CLI_NAME + exe.suffix)
            if console.is_file():
                return [str(console)]
        return [str(exe)]
    return _source_cli_base()


def _appimage() -> str | None:
    """The ``.AppImage`` file this frozen Linux build runs from, if any.

    Only trusted when frozen: ``APPIMAGE`` is inherited by every child of an
    AppImage (a terminal emulator AppImage, for instance).
    """
    appimage = os.environ.get("APPIMAGE", "")
    if sys.platform.startswith("linux") and appimage and os.path.isfile(appimage):
        return appimage
    return None


def _installer_command_on_path(exe: Path) -> bool:
    """Whether ``eye-tracker`` on this process's PATH is the Windows installer's
    ``eye-tracker.exe`` of this very copy (next to ``exe``).

    A process started before the installer changed the PATH (the app launched
    from its last page) does not see the entry yet and keeps the full path.
    """
    found = shutil.which(APP_SLUG)
    return found is not None and _same_file(found, exe.with_name(APP_SLUG + ".exe"))


def _program_name() -> str:
    """How ``--help`` names the program: :func:`cli_command` without directories.

    ``eye-tracker`` from source and on the PATH set by the Windows installer,
    ``eye-tracker-cli.exe`` or the ``.AppImage`` file name in the other packages.
    """
    base = _cli_base()
    if len(base) > 1:  # python -m eye_tracker
        return format_command([Path(base[0]).stem, *base[1:]])
    name = Path(base[0]).name
    if not paths.is_frozen() and Path(name).stem.lower() == APP_SLUG:
        return APP_SLUG  # the script, with or without .exe
    return name


def _source_cli_base() -> list[str]:
    """The ``eye-tracker`` script of a source or pip install (see :func:`cli_command`)."""
    if not sys.executable:  # an embedded interpreter: nothing better to name
        return [APP_SLUG]
    interpreter = Path(sys.executable)
    # Virtual environments and pip keep console scripts next to the interpreter.
    script = interpreter.with_name(APP_SLUG + (".exe" if sys.platform == "win32" else ""))
    have_script = script.is_file()
    on_path = shutil.which(APP_SLUG)
    if on_path is not None and (not have_script or _same_file(on_path, script)):
        return [APP_SLUG]
    if have_script:
        return [str(script)]
    if sys.platform == "win32" and interpreter.name.lower() == "pythonw.exe":
        # pythonw has no console: the command's output would be lost.
        console = interpreter.with_name("python.exe")
        if console.is_file():
            interpreter = console
    return [str(interpreter), "-m", "eye_tracker"]


def _same_file(a: str | Path, b: str | Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


# ------------------------------------------------------------------ entry points
def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point. Returns the process exit code."""
    _prepare_std_streams()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, --version and usage errors
        return _exit_code(exc)
    if args.config_dir:
        paths.set_base_override(Path(args.config_dir))
    command = args.command or "run"
    handler = _HANDLERS[command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED


def gui_main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the windowed launcher (``eye-tracker-gui``, ``pythonw`` and the
    windowed frozen builds).

    Such a process has no console: ``sys.stdout``/``sys.stderr`` may be ``None``
    (``pythonw`` on Windows), so they are pointed at ``os.devnull`` before anything
    can print. Everything else is the same as :func:`main`.
    """
    return main(argv)


# -------------------------------------------------------------------- commands
def _cmd_run(args: argparse.Namespace) -> int:
    from .app import run_app

    return run_app(args)


def _cmd_calibrate(args: argparse.Namespace) -> int:
    # run_app forwards "calibrate" to a running instance instead of "show", so this
    # covers both "open it in the running app" and "start the app and calibrate".
    args.calibrate = True
    from .app import run_app

    return run_app(args)


def _cmd_doctor(args: argparse.Namespace) -> int:
    _setup_cli_logging(args)
    from .diagnostics import collect_report, format_report

    report = collect_report(probe_cameras=args.probe_cameras)
    if args.json:
        _out(json.dumps(report, indent=2, default=str))
    else:
        _out(format_report(report), end="")
    return EXIT_OK


def _cmd_bench(args: argparse.Namespace) -> int:
    _setup_cli_logging(args)
    from .diagnostics import BenchError, format_bench, run_bench
    from .vision.backends import BackendUnavailable
    from .vision.camera import CameraError

    settings = _read_settings()
    device = args.camera if args.camera is not None else settings.camera.device
    backend = args.backend or settings.general.backend
    _err(f"Benchmarking for 2 x {args.seconds:g} s...")
    try:
        result = run_bench(
            args.seconds,
            device,
            backend,
            width=settings.camera.width,
            height=settings.camera.height,
            api=settings.camera.api,
        )
    except (BenchError, CameraError, BackendUnavailable) as exc:
        message = str(exc)
        if device.strip().isdigit() and _instance_running():
            message += (
                f"\n{APP_NAME} is running and may be holding the camera; "
                f"release it with '{cli_command_text('ctl', 'privacy-on')}' and try again."
            )
        _err(f"error: {message}")
        return EXIT_ERROR
    if args.json:
        _out(json.dumps(result, indent=2))
    else:
        _out(format_bench(result), end="")
    return EXIT_OK


def _cmd_ctl(args: argparse.Namespace) -> int:
    _setup_cli_logging(args)
    from . import ipc

    reply = ipc.send_command(args.action, timeout_ms=max(1, args.timeout))
    if reply is None:
        if ipc.is_running():
            _err(f"error: {APP_NAME} is running but did not answer within {args.timeout} ms.")
            return EXIT_ERROR
        _err(f"{APP_NAME} is not running. Start it with '{format_command(app_command())}'.")
        return EXIT_NOT_RUNNING
    if reply.startswith("error"):
        _err(reply)
        return EXIT_ERROR
    _out(reply)
    return EXIT_OK


def _cmd_autostart(args: argparse.Namespace) -> int:
    _setup_cli_logging(args)
    from .platform import autostart

    # The login entry starts this profile: with --config-dir it must pass the
    # same directory, or the default profile would start at login instead.
    profile = Path(args.config_dir) if args.config_dir else None
    if args.action == "status":
        if not autostart.is_supported():
            _out("Start at login: not supported on this system")
            return EXIT_OK
        status = autostart.status(config_dir=profile)
        # The value itself (enabled, disabled, stale, other-profile) is what
        # scripts and bug reports quote; the explanation says what to do.
        _out(f"Start at login: {status.value}")
        explanation = _AUTOSTART_STATUS_HELP.get(status.value)
        if explanation:
            _out(f"  {explanation.format(command=cli_command_text('autostart', 'enable'))}")
        _out(f"Entry:      {autostart.location()}")
        registered = autostart.registered_command()
        if registered:
            _out(f"Registered: {format_command(registered)}")
        _out(f"This copy:  {format_command(autostart.launch_command(config_dir=profile))}")
        return EXIT_OK
    if not autostart.is_supported():
        _err("error: start at login is not supported on this system.")
        return EXIT_ERROR
    try:
        if args.action == "enable":
            autostart.enable(background=True, config_dir=profile)
            _out(f"Start at login enabled ({autostart.location()}).")
        else:
            autostart.disable(config_dir=profile)
            if autostart.status(config_dir=profile) is autostart.Status.OTHER_PROFILE:
                _out("Start at login starts another profile; left unchanged.")
            else:
                _out("Start at login disabled.")
    except autostart.AutostartError as exc:
        _err(f"error: {exc}")
        return EXIT_ERROR
    return EXIT_OK


#: What ``autostart status`` adds for a :class:`~eye_tracker.platform.autostart.Status`
#: value that needs an explanation (``{command}``: this copy's ``autostart enable``).
_AUTOSTART_STATUS_HELP = {
    "stale": "Broken: the registered program no longer exists or is in a temporary "
    "location. Run '{command}' to repair it.",
    "other-profile": "The entry starts another profile (--config-dir). Run "
    "'{command}' to start this one instead.",
}


def _cmd_reset(args: argparse.Namespace) -> int:
    _setup_cli_logging(args)
    targets: list[Path] = []
    if args.all or args.settings:
        settings_file = paths.settings_file()
        # Settings.load moves an unreadable file aside under this name.
        targets += [settings_file, settings_file.with_suffix(settings_file.suffix + ".corrupt")]
    if args.all or args.calibration:
        targets.append(paths.calibration_file())
    if not targets:
        _err("error: choose what to reset: --calibration, --settings or --all")
        return EXIT_USAGE
    if _instance_running():
        _err(
            f"error: {APP_NAME} is running and would write its data back. "
            f"Quit it first ('{cli_command_text('ctl', 'quit')}'), then run reset again."
        )
        return EXIT_ERROR
    removed = 0
    for path in targets:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            _err(f"error: could not delete {path}: {exc}")
            return EXIT_ERROR
        _out(f"Deleted {path}")
        removed += 1
    if not removed:
        _out("Nothing to reset.")
    return EXIT_OK


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": _cmd_run,
    "calibrate": _cmd_calibrate,
    "doctor": _cmd_doctor,
    "bench": _cmd_bench,
    "ctl": _cmd_ctl,
    "autostart": _cmd_autostart,
    "reset": _cmd_reset,
}


# ---------------------------------------------------------------------- helpers
def _prepare_std_streams() -> None:
    """Make printing safe without a console and on legacy code pages."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is None:
            # pythonw and windowed builds have no console; argparse would crash.
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(errors="replace")


def _exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    return EXIT_ERROR


def _setup_cli_logging(args: argparse.Namespace) -> None:
    # Command-line tools report on the terminal and leave the app's log file alone.
    setup_logging(args.log_level or "WARNING", console=True, log_to_file=False)


def _read_settings() -> Settings:
    """Settings as the app would see them, without moving a corrupt file aside."""
    path = paths.settings_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Settings()
    except (OSError, ValueError) as exc:
        log.warning("Ignoring unreadable settings %s: %s", path, exc)
        return Settings()
    return Settings.from_dict(data)


def _instance_running() -> bool:
    from . import ipc

    try:
        return ipc.is_running()
    except Exception:
        log.debug("Could not check for a running instance", exc_info=True)
        return False


def _out(message: str, end: str = "\n") -> None:
    sys.stdout.write(message + end)
    sys.stdout.flush()


def _err(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()
