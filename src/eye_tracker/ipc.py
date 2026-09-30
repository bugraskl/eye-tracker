"""Single-instance guard and local command channel.

A running app listens on a per-user local socket (a named pipe on Windows, a
Unix domain socket elsewhere). Launching the app a second time, ``eye-tracker
ctl <command>`` and desktop keyboard shortcuts (the only way to bind hotkeys on
Wayland) all talk to it through this channel.

Protocol: the client sends one UTF-8 line with a command from :data:`COMMANDS`
and receives one line back: ``ok``, ``error: <reason>`` or, for ``status``, a
JSON object. The server then closes the connection.

This is the only networking in the app, and it is local-only: Qt's local
sockets never touch the network, and the socket is restricted to the current
user (``QLocalServer.UserAccessOption``).
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable

from PySide6.QtCore import QCoreApplication, QEventLoop, QObject, QTimer, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from . import paths

__all__ = [
    "COMMANDS",
    "DEFAULT_TIMEOUT_MS",
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

ReplyCallable = Callable[[str], None]
CommandHandler = Callable[[str], str]

# A QCoreApplication created by this module for command-line use (kept alive).
_owned_app: list[QCoreApplication] = []


def server_name() -> str:
    """Name (or, on Linux, full socket path) of this user's instance socket.

    On Linux the socket goes into ``$XDG_RUNTIME_DIR`` (private to the user and
    cleaned at logout) instead of the shared ``/tmp`` Qt would use otherwise.
    """
    name = paths.ipc_name()
    if sys.platform.startswith("linux"):
        runtime = os.environ.get("XDG_RUNTIME_DIR", "")
        if runtime and os.path.isabs(runtime) and os.path.isdir(runtime):
            full = os.path.join(runtime, name)
            if len(full.encode("utf-8")) <= _MAX_SOCKET_PATH:
                return full
    return name


def send_command(command: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str | None:
    """Send ``command`` to the running instance and return its one-line reply.

    Returns ``None`` when no instance is running, or when one accepted the
    connection but did not answer within ``timeout_ms``. Works with only a
    ``QCoreApplication``; one is created if no Qt application exists yet (so
    create your own ``QApplication`` *before* calling this if you need one).
    """
    text = command.strip().lower()
    if not text or "\n" in text or "\r" in text:
        raise ValueError(f"invalid command {command!r}")
    _allow_foreground_handoff()
    connected, data = _exchange(server_name(), (text + "\n").encode("utf-8"), timeout_ms)
    if not connected:
        return None
    if not data:
        log.warning("The running instance did not answer %r within %d ms", text, timeout_ms)
        return None
    line = data.split(b"\n", 1)[0]
    return line.decode("utf-8", errors="replace").rstrip("\r")


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


class InstanceServer(QObject):
    """Local socket server of the running instance.

    Each request is announced through :attr:`command_received` as
    ``(command, reply)``, where ``reply(text)`` sends the answer (only the first
    call counts). If ``handler`` is given and no slot has replied synchronously,
    ``handler(command)`` is called and its return value is sent as the reply.
    Unknown commands are rejected before either is involved.

    Args:
        handler: Synchronous command handler returning the reply line.
        parent: Qt parent.
        name: Server name override (tests); defaults to :func:`server_name`.
    """

    command_received = Signal(str, object)

    def __init__(
        self,
        handler: CommandHandler | None = None,
        parent: QObject | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._handler = handler
        self._name = name or server_name()
        self._server: QLocalServer | None = None
        self._connections: set[_Connection] = set()
        self._another_instance = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_listening(self) -> bool:
        return self._server is not None and self._server.isListening()

    @property
    def another_instance_running(self) -> bool:
        """True when :meth:`listen` failed because another instance owns the socket."""
        return self._another_instance

    def listen(self) -> bool:
        """Start listening. Returns ``False`` if another instance is already listening
        (see :attr:`another_instance_running`) or the socket cannot be created."""
        if self.is_listening:
            return True
        self._another_instance = False
        # Windows lets two servers listen on the same pipe name, so a live
        # instance must be detected by connecting to it rather than by failure.
        if _exchange(self._name, None, PROBE_TIMEOUT_MS)[0]:
            self._another_instance = True
            log.info("Another instance is already listening on %s", self._name)
            return False
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        if not server.listen(self._name):
            # An instance that crashed leaves its socket file behind on Unix;
            # nobody answered the probe, so it is stale and safe to remove.
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
        """Stop listening and drop open connections. Safe to call more than once."""
        server, self._server = self._server, None
        if server is not None:
            server.close()
            server.deleteLater()
        for connection in list(self._connections):
            connection.abort()
        self._connections.clear()

    # ---------------------------------------------------------------- internals
    def _on_new_connection(self) -> None:
        server = self._server
        if server is None:
            return
        while server.hasPendingConnections():
            socket = server.nextPendingConnection()
            if socket is not None:
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
    state = {"connected": False, "done": False}

    def finish() -> None:
        state["done"] = True
        if loop.isRunning():
            loop.quit()

    def drain() -> None:
        # Reading a socket that never connected makes Qt warn "device not open".
        if socket.isOpen():
            buffer.extend(socket.readAll().data())

    def on_connected() -> None:
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
    return state["connected"], bytes(buffer)
