"""Single-instance guard and local command channel.

A running app listens on a per-user local socket (a named pipe on Windows, a
Unix domain socket elsewhere). Launching the app a second time, ``eye-tracker
ctl <command>`` and desktop keyboard shortcuts (the only way to bind hotkeys on
Wayland) all talk to it through this channel.

Which process *is* the running instance is decided by :class:`InstanceLock`,
an OS file lock, not by the socket: two servers can listen on one name (Windows
creates a second pipe instance, Qt on Unix renames its socket over the old
one), so "nobody answered, so I listen" is a race between two launches.

Protocol: the client sends one UTF-8 line with a command from :data:`COMMANDS`
and receives one line back: ``ok``, ``error: <reason>`` or, for ``status``, a
JSON object. The server then closes the connection.

This is the only networking in the app, and it is local-only: Qt's local
sockets never touch the network, and the socket is restricted to the current
user (``QLocalServer.UserAccessOption``). The socket name is predictable, so a
client also checks that the server that answered runs under the same user
account before it sends anything (see :func:`_server_is_trusted`).

On Windows that restriction is not enough by itself: pipe names are
machine-wide and a pipe keeps the security descriptor of whoever created the
name *first*, so another account that creates our name before we listen would
decide who may connect. The server therefore refuses to listen on a name that
already exists while it holds the :class:`InstanceLock`, and it also checks
the account of every client before reading a request (see
:func:`_client_is_trusted`). On Unix the socket lives in a directory only this
user can enter, which already keeps other accounts out.
"""

from __future__ import annotations

import errno
import functools
import logging
import os
import stat
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PySide6.QtCore import QCoreApplication, QEventLoop, QObject, QTimer, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from . import APP_SLUG, paths

__all__ = [
    "COMMANDS",
    "DEFAULT_TIMEOUT_MS",
    "STARTUP_WAIT_MS",
    "InstanceLock",
    "InstanceServer",
    "is_running",
    "send_command",
    "server_name",
]

log = logging.getLogger(__name__)

#: Commands understood by a running instance.
COMMANDS: frozenset[str] = frozenset(
    {
        "show",
        "settings",
        "pause",
        "resume",
        "toggle",
        "privacy-on",
        "privacy-off",
        "privacy-toggle",
        "calibrate",
        "status",
        "quit",
    }
)

DEFAULT_TIMEOUT_MS = 1500
#: How long a launch that lost the :class:`InstanceLock` keeps asking the
#: winner, which takes the lock before it starts listening.
STARTUP_WAIT_MS = 5000
#: Pause between two attempts while waiting for an instance to answer.
RETRY_INTERVAL_MS = 250
#: How long :func:`is_running` and the pre-listen probe wait for a connection.
PROBE_TIMEOUT_MS = 300
#: A client that has not sent a complete request by then is disconnected.
CLIENT_IDLE_TIMEOUT_MS = 3000
#: Requests are single short words; anything longer is garbage or abuse.
MAX_REQUEST_BYTES = 1024
#: Replies are one line; ``status`` JSON stays far below this.
MAX_REPLY_BYTES = 64 * 1024
# Unix socket paths are limited to ~104-108 bytes depending on the OS.
_MAX_SOCKET_PATH = 100
#: How long a pipe name that exists before we listen may take to disappear
#: (Windows): a client still holding a pipe of an instance that just exited
#: keeps the name alive for a moment. A name that stays belongs to someone else.
PIPE_NAME_GRACE_S = 1.0
_PIPE_NAME_POLL_S = 0.1
_PIPE_PREFIX = "\\\\.\\pipe\\"

ReplyCallable = Callable[[str], None]
CommandHandler = Callable[[str], str]

# A QCoreApplication created by this module for command-line use (kept alive).
_owned_app: list[QCoreApplication] = []


def server_name() -> str:
    """Name (or, on Linux, full socket path) of this user's instance socket.

    On Linux the socket goes into ``$XDG_RUNTIME_DIR`` (private to the user and
    cleaned at logout) or, without one, into a private ``/tmp/eye-tracker-<uid>``
    directory, instead of the shared ``/tmp`` Qt would use, where another
    account could create the socket first.
    """
    name = paths.ipc_name()
    if sys.platform.startswith("linux"):
        directory = _private_runtime_dir()
        if directory:
            full = os.path.join(directory, name)
            if len(full.encode("utf-8")) <= _MAX_SOCKET_PATH:
                return full
    return name


def _private_runtime_dir() -> str | None:
    """A directory only this user can create files in (Linux socket location)."""
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if runtime and os.path.isabs(runtime) and os.path.isdir(runtime):
        return runtime
    return _private_temp_dir()


def _private_temp_dir() -> str | None:
    """``<tmp>/eye-tracker-<uid>``, created with mode 0700.

    ``None`` when it cannot be created or is not safely ours: another account
    may have created it first, which must not let it choose our socket.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return None
    uid = getuid()
    path = os.path.join(tempfile.gettempdir(), f"{APP_SLUG}-{uid}")
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError:
        log.debug("Cannot create %s", path, exc_info=True)
        return None
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or info.st_mode & 0o077:
        log.warning("Not using %s for the instance socket: it is not a private directory", path)
        return None
    return path


def send_command(
    command: str, timeout_ms: int = DEFAULT_TIMEOUT_MS, *, wait_ms: int = 0
) -> str | None:
    """Send ``command`` to the running instance and return its one-line reply.

    Returns ``None`` when no instance is running, or when one accepted the
    connection but did not answer within ``timeout_ms``. Works with only a
    ``QCoreApplication``; one is created if no Qt application exists yet (so
    create your own ``QApplication`` *before* calling this if you need one).

    ``wait_ms`` keeps trying for that long while nothing answers: a launch that
    lost the :class:`InstanceLock` uses it (with :data:`STARTUP_WAIT_MS`) to
    reach an instance that holds the lock but is still starting up. A server
    that accepts the connection but never replies is not asked twice.
    """
    text = command.strip().lower()
    if not text or "\n" in text or "\r" in text:
        raise ValueError(f"invalid command {command!r}")
    request = (text + "\n").encode("utf-8")
    deadline = time.monotonic() + max(0, wait_ms) / 1000.0
    while True:
        _allow_foreground_handoff()
        connected, data = _exchange(server_name(), request, timeout_ms)
        if connected:
            break
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            return None
        _pause(min(RETRY_INTERVAL_MS, remaining_ms))
    if not data:
        log.warning("The running instance did not answer %r within %d ms", text, timeout_ms)
        return None
    line = data.split(b"\n", 1)[0]
    return line.decode("utf-8", errors="replace").rstrip("\r")


def _pause(ms: int) -> None:
    """Wait ``ms`` while still processing events (a server may live in this thread)."""
    _ensure_core_app()
    loop = QEventLoop()
    QTimer.singleShot(max(1, int(ms)), loop.quit)
    loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)


def _allow_foreground_handoff() -> None:
    """Let the running instance bring a window to the front (Windows).

    Windows only lets the process the user is interacting with take the
    foreground. The client (a terminal running ``ctl``, or a second launch from
    the Start menu) is that process, so it passes the right on; otherwise
    "show"/"settings" would only flash the taskbar button.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.AllowSetForegroundWindow.argtypes = [ctypes.c_uint32]
        user32.AllowSetForegroundWindow.restype = ctypes.c_int
        user32.AllowSetForegroundWindow(0xFFFFFFFF)  # ASFW_ANY
    except Exception:
        log.debug("AllowSetForegroundWindow failed", exc_info=True)


def is_running(timeout_ms: int = PROBE_TIMEOUT_MS) -> bool:
    """Whether an instance for this user (and config directory) is listening."""
    connected, _ = _exchange(server_name(), None, timeout_ms)
    return connected


# ------------------------------------------------------------------------ lock
class InstanceLock:
    """Exclusive lock that makes one process *the* running instance.

    An OS file lock (``flock`` on Unix, a byte-range lock on Windows) on a file
    named after the socket, so every ``--config-dir`` profile has its own.
    Taking it is atomic, so of two launches racing each other exactly one wins,
    and the OS drops it when the holder exits or crashes, so it never goes
    stale. Hold it for the life of the process (:class:`InstanceServer`
    releases it in :meth:`~InstanceServer.close`).

    The file lives on a local disk, next to the socket where that is a file:
    a lock in a home directory on a network share would also stop the same
    user's tracker on *another* computer.

    Args:
        name: Socket name the lock belongs to; defaults to :func:`server_name`.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = name or server_name()
        self._fd: int | None = None
        self._enforced = True

    @property
    def path(self) -> Path:
        """The lock file."""
        if os.path.isabs(self._name):  # a socket path (Linux): keep them together
            return Path(self._name + ".lock")
        if sys.platform == "darwin":
            # Qt puts the socket into $TMPDIR, which is private to the user and
            # always local; the config directory may be in a network home.
            return Path(tempfile.gettempdir()) / f"{self._name}.lock"
        # Windows: pipes have no file. The config directory is the local (not
        # roaming) AppData one, or the --config-dir the name is derived from.
        return paths.config_dir() / f"{self._name}.lock"

    @property
    def is_held(self) -> bool:
        """True while this object holds the lock."""
        return self._fd is not None

    @property
    def enforced(self) -> bool:
        """False after :meth:`acquire` found that this system cannot lock the file."""
        return self._enforced

    def acquire(self) -> bool:
        """Take the lock without waiting.

        Returns ``False`` only when another process (or another ``InstanceLock``
        in this one) holds it. If the file cannot be locked at all (an odd
        filesystem), logs a warning, sets :attr:`enforced` to ``False`` and
        returns ``True``: callers then fall back to probing the socket.
        """
        if self._fd is not None:
            return True
        try:
            path = self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            log.warning("Cannot create the instance lock (%s); relying on the socket alone", exc)
            self._enforced = False
            return True
        try:
            locked = _try_lock(fd)
        except OSError as exc:
            os.close(fd)
            log.warning("Cannot lock %s (%s); relying on the socket alone", path, exc)
            self._enforced = False
            return True
        if not locked:
            os.close(fd)
            return False
        self._fd = fd
        self._enforced = True
        log.debug("Holding the instance lock %s", path)
        return True

    def release(self) -> None:
        """Give the lock up. Safe to call when it is not held.

        The file itself stays: deleting it would let a later launch lock a new
        file while a slower one still locks the old, unlinked one.
        """
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        except OSError:
            log.debug("Unlocking the instance lock failed", exc_info=True)
        finally:
            os.close(fd)


def _try_lock(fd: int) -> bool:
    """Lock ``fd`` exclusively without waiting; ``False`` when someone else holds it.

    Raises ``OSError`` when the file cannot be locked at all.
    """
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            # Locking beyond the end of the (empty) file is allowed.
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EDEADLOCK):
                return False
            raise
        return True
    import fcntl

    try:
        # flock, not lockf: its locks also conflict between two descriptors of
        # one process, and they belong to the open file, not the process.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(fd: int) -> None:
    """Drop the lock right away (closing the file does it too, eventually on Windows)."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


# ---------------------------------------------------------------------- server
class InstanceServer(QObject):
    """Local socket server of the running instance.

    Each request is announced through :attr:`command_received` as
    ``(command, reply)``, where ``reply(text)`` sends the answer (only the first
    call counts). If ``handler`` is given and no slot has replied synchronously,
    ``handler(command)`` is called and its return value is sent as the reply.
    Unknown commands are rejected before either is involved.

    Only the holder of the :class:`InstanceLock` listens: :meth:`listen` takes
    the lock (or uses one the caller already took) and :meth:`close` releases it.

    Args:
        handler: Synchronous command handler returning the reply line.
        parent: Qt parent.
        name: Server name override (tests); defaults to :func:`server_name`.
        lock: The instance lock, typically acquired at startup before anything
            else; by default :meth:`listen` creates and takes one for ``name``.
    """

    command_received = Signal(str, object)

    def __init__(
        self,
        handler: CommandHandler | None = None,
        parent: QObject | None = None,
        *,
        name: str | None = None,
        lock: InstanceLock | None = None,
    ) -> None:
        super().__init__(parent)
        self._handler = handler
        self._name = name or server_name()
        self._lock = lock
        self._server: QLocalServer | None = None
        self._connections: set[_Connection] = set()
        self._another_instance = False
        #: Connections refused by the account check (only the first is logged loudly).
        self._refused_clients = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def lock(self) -> InstanceLock | None:
        """The instance lock (created by :meth:`listen` unless one was passed in)."""
        return self._lock

    @property
    def is_listening(self) -> bool:
        return self._server is not None and self._server.isListening()

    @property
    def another_instance_running(self) -> bool:
        """True when :meth:`listen` failed because another instance holds the lock
        (or, where files cannot be locked, answers on the socket)."""
        return self._another_instance

    def listen(self) -> bool:
        """Become the running instance and start listening.

        Returns ``False`` if another instance holds the :class:`InstanceLock`
        (see :attr:`another_instance_running`) or the socket cannot be created;
        in the latter case the lock stays held, since this process still is the
        instance, just one that ``ctl`` cannot reach. That includes a Windows
        pipe name some other process created first (see the module docs).
        """
        if self.is_listening:
            return True
        self._another_instance = False
        if self._lock is None:
            self._lock = InstanceLock(self._name)
        lock = self._lock
        if not lock.acquire():
            self._another_instance = True
            log.info("Another instance holds %s", lock.path)
            return False
        # Without a file lock, fall back to asking the socket. (Two servers can
        # listen on one name, so listen() itself would not fail.)
        if not lock.is_held and _exchange(self._name, None, PROBE_TIMEOUT_MS)[0]:
            self._another_instance = True
            log.info("Another instance is already listening on %s", self._name)
            return False
        if _pipe_name_taken(self._name):
            # Our own instance would have answered the probe or held the lock,
            # so this is another account's pipe. Qt would happily add instances
            # to it, and they would carry that account's access rules.
            from .cli import cli_command_text  # named as this copy is run

            log.warning(
                "The command channel %s already exists and does not belong to this instance "
                "(most likely another user account created it); '%s' and "
                "desktop shortcuts cannot reach this instance",
                self._name,
                cli_command_text("ctl"),
            )
            return False
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        if not server.listen(self._name):
            # An instance that crashed leaves its socket file behind on Unix. We
            # hold the lock (or nobody answered the probe), so it is stale.
            log.debug(
                "listen(%s) failed (%s); removing a stale socket", self._name, server.errorString()
            )
            QLocalServer.removeServer(self._name)
            if not server.listen(self._name):
                log.warning(
                    "Cannot create the instance socket %s: %s", self._name, server.errorString()
                )
                server.deleteLater()
                return False
        server.newConnection.connect(self._on_new_connection)
        self._server = server
        log.debug("Listening for commands on %s", server.fullServerName())
        return True

    def close(self) -> None:
        """Stop listening, drop open connections and release the instance lock.

        Safe to call more than once.
        """
        server, self._server = self._server, None
        if server is not None:
            server.close()
            server.deleteLater()
        for connection in list(self._connections):
            connection.abort()
        self._connections.clear()
        if self._lock is not None:
            self._lock.release()

    # ---------------------------------------------------------------- internals
    def _on_new_connection(self) -> None:
        server = self._server
        if server is None:
            return
        while server.hasPendingConnections():
            socket = server.nextPendingConnection()
            if socket is None:
                continue
            if not _client_is_trusted(socket):
                # Checked before a byte is read: pause, privacy-on or quit from
                # another account would switch off the walk-away lock.
                level = logging.DEBUG if self._refused_clients else logging.WARNING
                self._refused_clients += 1
                log.log(level, "Refused a command connection from another user account")
                socket.abort()
                socket.deleteLater()
                continue
            self._connections.add(_Connection(self, socket))

    def _forget(self, connection: _Connection) -> None:
        self._connections.discard(connection)

    def _dispatch(self, raw: bytes, connection: _Connection) -> None:
        try:
            command = raw.decode("utf-8").strip().lower()
        except UnicodeDecodeError:
            connection.reply("error: the request is not valid UTF-8")
            return
        if command not in COMMANDS:
            valid = ", ".join(sorted(COMMANDS))
            connection.reply(f"error: unknown command {command!r} (valid: {valid})")
            return
        log.info("Received command %r", command)
        try:
            self.command_received.emit(command, connection.reply)
        except Exception:
            log.exception("A command_received slot failed for %r", command)
        if connection.replied or self._handler is None:
            # Either answered already, or a slot will reply later (the idle
            # timeout closes the connection if nobody ever does).
            return
        try:
            result = self._handler(command)
        except Exception as exc:
            log.exception("Command %r failed", command)
            result = f"error: {exc}"
        connection.reply(result if isinstance(result, str) else "ok")


class _Connection:
    """One client connection: buffers the request line and sends a single reply."""

    def __init__(self, owner: InstanceServer, socket: QLocalSocket) -> None:
        self._owner = owner
        self._socket = socket
        self._buffer = bytearray()
        self.replied = False
        self._dispatched = False
        self._timer = QTimer(socket)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._timer.start(CLIENT_IDLE_TIMEOUT_MS)
        socket.readyRead.connect(self._on_ready_read)
        socket.disconnected.connect(self._on_disconnected)
        # Data may already be buffered before the slots were connected.
        if socket.bytesAvailable() > 0:
            self._on_ready_read()

    def reply(self, text: str) -> None:
        """Send ``text`` as the one-line reply and close. Later calls are ignored."""
        if self.replied:
            return
        self.replied = True
        self._timer.stop()
        socket = self._socket
        if socket.state() != QLocalSocket.LocalSocketState.ConnectedState:
            return
        line = " ".join(str(text).splitlines()) or "ok"
        socket.write((line + "\n").encode("utf-8"))
        socket.flush()
        # Closes once the reply has been written.
        socket.disconnectFromServer()

    def abort(self) -> None:
        """Close now; a reply already queued (e.g. to ``quit``) gets a moment to go out."""
        self.replied = True
        self._timer.stop()
        socket = self._socket
        if (
            socket.state() != QLocalSocket.LocalSocketState.UnconnectedState
            and socket.bytesToWrite() > 0
        ):
            socket.waitForBytesWritten(200)
        socket.abort()

    def _on_ready_read(self) -> None:
        data = self._socket.readAll().data()
        if self.replied or self._dispatched:
            return
        self._buffer.extend(data)
        newline = self._buffer.find(b"\n")
        if newline < 0:
            if len(self._buffer) > MAX_REQUEST_BYTES:
                self.reply("error: request too long")
            return
        # One request per connection; anything after the first line is ignored.
        self._dispatched = True
        self._owner._dispatch(bytes(self._buffer[:newline]), self)

    def _on_timeout(self) -> None:
        if not self.replied:
            log.debug("Closing an idle command connection")
            self.reply(
                "error: no complete request received"
                if not self._buffer
                else "error: the command was not answered in time"
            )

    def _on_disconnected(self) -> None:
        self.replied = True
        self._timer.stop()
        self._owner._forget(self)
        self._socket.deleteLater()


# ---------------------------------------------------------------------- client
def _ensure_core_app() -> None:
    """Local sockets need a Qt application object for their event handling."""
    if QCoreApplication.instance() is None:
        program = sys.argv[0] if sys.argv and sys.argv[0] else "eye-tracker"
        _owned_app.append(QCoreApplication([program]))


def _exchange(name: str, request: bytes | None, timeout_ms: int) -> tuple[bool, bytes]:
    """Connect to ``name``; send ``request`` and read until a newline, close or timeout.

    With ``request=None`` only the connection is tested. Returns
    ``(connected, data)``.

    A local event loop is used instead of the blocking ``waitFor*`` calls: those
    hold the GIL while waiting, and they cannot reach a server living in the
    same thread (tests, or a probe made by the process that is about to listen).
    """
    _ensure_core_app()
    socket = QLocalSocket()
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    buffer = bytearray()
    state = {"connected": False, "done": False, "rejected": False}

    def finish() -> None:
        state["done"] = True
        if loop.isRunning():
            loop.quit()

    def drain() -> None:
        # Reading a socket that never connected makes Qt warn "device not open".
        if socket.isOpen() and not state["rejected"]:
            buffer.extend(socket.readAll().data())

    def on_connected() -> None:
        # Checked before a single byte is sent: an impostor learns nothing and,
        # on Windows, cannot impersonate us (that needs data read from the pipe).
        if not _server_is_trusted(socket):
            state["rejected"] = True
            finish()
            return
        state["connected"] = True
        if request is None:
            finish()
            return
        socket.write(request)
        socket.flush()

    def on_ready_read() -> None:
        drain()
        if b"\n" in buffer or len(buffer) > MAX_REPLY_BYTES:
            finish()

    def on_closed(*_args: object) -> None:
        drain()
        finish()

    socket.connected.connect(on_connected)
    socket.readyRead.connect(on_ready_read)
    socket.disconnected.connect(on_closed)
    socket.errorOccurred.connect(on_closed)
    timer.timeout.connect(finish)

    socket.connectToServer(name)
    # Connecting to a missing server fails synchronously; so may everything else
    # on some platforms, in which case the loop must not start at all (a quit()
    # issued before exec() would be lost and the loop would wait for the timeout).
    if socket.state() == QLocalSocket.LocalSocketState.UnconnectedState:
        state["done"] = True
    if not state["done"]:
        timer.start(max(1, int(timeout_ms)))
        loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    timer.stop()
    # The socket and timer are parentless and freed with their Python wrappers.
    socket.blockSignals(True)
    socket.abort()
    if state["rejected"]:
        log.warning("Ignoring the instance socket %s: another user account owns it", name)
    return state["connected"], bytes(buffer)


# ------------------------------------------------------------ server identity
def _server_is_trusted(socket: QLocalSocket) -> bool:
    """False when the server that answered on our socket name runs as another user.

    The name is predictable and, on Windows, pipe names are machine-wide, so
    another local account could create it first: to keep this app from
    starting (and its walk-away lock from running), or to read our requests.
    ``True`` when the owner cannot be determined (nothing to go on).
    """
    try:
        if sys.platform == "win32":
            verdict = _pipe_server_is_current_user(int(socket.socketDescriptor()))
        else:
            verdict = _socket_file_is_current_user(socket.fullServerName())
    except Exception:
        log.debug("Could not identify the owner of the instance socket", exc_info=True)
        verdict = None
    return verdict is not False


def _client_is_trusted(socket: QLocalSocket) -> bool:
    """False when the process that connected to our server runs as another user.

    Only Windows needs this (see the module docs): a pipe name another account
    created first keeps that account's access rules, so its owner could connect
    to our instances and pause tracking, switch privacy mode on or quit. Elevated
    processes of the same user pass. ``True`` when the client cannot be
    identified, like :func:`_server_is_trusted`, and always on Unix, where the
    private socket directory is the access check.
    """
    if sys.platform != "win32":
        return True
    try:
        verdict = _pipe_client_is_current_user(int(socket.socketDescriptor()))
    except Exception:
        log.debug("Could not identify the command client", exc_info=True)
        verdict = None
    return verdict is not False


def _socket_file_is_current_user(path: str) -> bool | None:
    """Whether the Unix socket file at ``path`` belongs to this user.

    The socket file is created by the process that listens on it, so its owner
    is the server's user. (Another account can neither replace our file nor
    create one owned by us.)
    """
    getuid = getattr(os, "getuid", None)
    if getuid is None or not path:
        return None
    try:
        owner = os.stat(path).st_uid
    except OSError:
        return None
    return owner == getuid()


#: ``OpenProcess`` right that other processes of the same user always grant.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1  # TOKEN_INFORMATION_CLASS.TokenUser


@functools.cache
def _win_security() -> Any:
    """ctypes prototypes for the Windows identity checks (``None`` elsewhere)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes as w

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    except OSError:
        return None

    def bind(dll: Any, name: str, restype: Any, *argtypes: Any) -> Any:
        fn = getattr(dll, name)
        fn.restype = restype
        fn.argtypes = list(argtypes)
        return fn

    return SimpleNamespace(
        server_pid=bind(
            kernel32, "GetNamedPipeServerProcessId", w.BOOL, w.HANDLE, ctypes.POINTER(w.ULONG)
        ),
        client_pid=bind(
            kernel32, "GetNamedPipeClientProcessId", w.BOOL, w.HANDLE, ctypes.POINTER(w.ULONG)
        ),
        wait_named_pipe=bind(kernel32, "WaitNamedPipeW", w.BOOL, w.LPCWSTR, w.DWORD),
        open_process=bind(kernel32, "OpenProcess", w.HANDLE, w.DWORD, w.BOOL, w.DWORD),
        close_handle=bind(kernel32, "CloseHandle", w.BOOL, w.HANDLE),
        current_process=bind(kernel32, "GetCurrentProcess", w.HANDLE),
        open_token=bind(
            advapi32, "OpenProcessToken", w.BOOL, w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)
        ),
        token_info=bind(
            advapi32,
            "GetTokenInformation",
            w.BOOL,
            w.HANDLE,
            ctypes.c_int,
            w.LPVOID,
            w.DWORD,
            ctypes.POINTER(w.DWORD),
        ),
        equal_sid=bind(advapi32, "EqualSid", w.BOOL, w.LPVOID, w.LPVOID),
    )


def _pipe_server_is_current_user(handle: int) -> bool | None:
    """Whether the process serving the pipe ``handle`` runs as this user (Windows)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes as w

    api = _win_security()
    if api is None or handle <= 0:
        return None
    pid = w.ULONG(0)
    if not api.server_pid(w.HANDLE(handle), ctypes.byref(pid)):
        return None
    if pid.value == os.getpid():
        return True
    return _process_is_current_user(int(pid.value))


def _pipe_client_is_current_user(handle: int) -> bool | None:
    """Whether the process on the client end of pipe ``handle`` runs as this user (Windows).

    The pid is recorded by the pipe file system when the client connects, so
    the client cannot choose it. A client that exited meanwhile cannot be
    opened and counts as another account, which only drops a dead connection.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes as w

    api = _win_security()
    if api is None or handle <= 0:
        return None
    pid = w.ULONG(0)
    if not api.client_pid(w.HANDLE(handle), ctypes.byref(pid)):
        return None
    if pid.value == os.getpid():
        return True
    return _process_is_current_user(int(pid.value))


#: ``WaitNamedPipeW`` results (``GetLastError``) that tell whether a pipe name exists.
_ERROR_FILE_NOT_FOUND = 2
_ERROR_SEM_TIMEOUT = 121  # the name exists, but no instance is waiting for a client
_ERROR_PIPE_BUSY = 231


def _pipe_name_taken(name: str) -> bool:
    """Whether a named pipe called ``name`` already exists (Windows; ``False`` elsewhere).

    ``WaitNamedPipeW`` only looks the name up; unlike opening the pipe it
    neither connects nor uses up an instance. A name that exists is watched
    for :data:`PIPE_NAME_GRACE_S`, since one whose last server just exited can
    linger while a client still holds it. Any answer other than "exists" or
    "not found" counts as free: refusing to listen would cost ``ctl`` for
    nothing.
    """
    if sys.platform != "win32":
        return False
    import ctypes

    api = _win_security()
    if api is None:
        return False
    path = name if name.startswith("\\\\") else _PIPE_PREFIX + name
    deadline = time.monotonic() + PIPE_NAME_GRACE_S
    while True:
        if api.wait_named_pipe(path, 1):
            exists = True
        else:
            error = ctypes.get_last_error()
            exists = error in (_ERROR_SEM_TIMEOUT, _ERROR_PIPE_BUSY)
            if not exists and error != _ERROR_FILE_NOT_FOUND:
                log.debug("WaitNamedPipe(%s) failed with error %d", path, error)
        if not exists:
            return False
        if time.monotonic() >= deadline:
            return True
        time.sleep(_PIPE_NAME_POLL_S)


def _process_is_current_user(pid: int) -> bool | None:
    """Whether process ``pid`` runs under this process's user account (Windows).

    ``False`` also when the process cannot be opened: processes of the same
    user always grant limited query access, so that means another account.
    """
    if sys.platform != "win32":
        return None
    api = _win_security()
    own = _own_token_user()
    if api is None or own is None:
        return None
    process = api.open_process(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not process:
        return False
    try:
        theirs = _token_user(process)
    finally:
        api.close_handle(process)
    if theirs is None:
        return False
    return bool(api.equal_sid(_sid_pointer(own), _sid_pointer(theirs)))


@functools.cache
def _own_token_user() -> Any:
    """``TOKEN_USER`` of this process (cached; it cannot change)."""
    api = _win_security()
    return _token_user(api.current_process()) if api is not None else None


def _token_user(process: Any) -> Any:
    """The ``TOKEN_USER`` structure of ``process`` as a ctypes buffer, or ``None``."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes as w

    api = _win_security()
    token = w.HANDLE()
    if api is None or not api.open_token(process, _TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        size = w.DWORD(0)
        api.token_info(token, _TOKEN_USER, None, 0, ctypes.byref(size))  # asks for the size
        if not size.value:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if not api.token_info(token, _TOKEN_USER, buffer, size, ctypes.byref(size)):
            return None
        return buffer
    finally:
        api.close_handle(token)


def _sid_pointer(token_user: Any) -> int | None:
    """``TOKEN_USER.User.Sid``: the first field of the structure, a pointer into it."""
    import ctypes

    return ctypes.c_void_p.from_buffer(token_user).value
