"""Tests for eye_tracker.ipc: the single-instance socket and command channel.

Server and client live in the same thread: the client runs a local event loop
while it waits, which also serves the server. Every test uses the ``app_dirs``
fixture, so the socket name (derived from the config directory) is unique.
"""

from __future__ import annotations

import errno
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QCoreApplication, QTimer

from eye_tracker import ipc
from eye_tracker.engine.controller import COMMANDS as CONTROLLER_COMMANDS


@pytest.fixture(autouse=True)
def private_runtime_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep Linux sockets and lock files out of the real ``$XDG_RUNTIME_DIR``."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))


@pytest.fixture
def server_factory(qapp: Any, app_dirs: Path) -> Iterator[Callable[..., ipc.InstanceServer]]:
    servers: list[ipc.InstanceServer] = []

    def make(handler: Callable[[str], str] | None = None, **kwargs: Any) -> ipc.InstanceServer:
        server = ipc.InstanceServer(handler, **kwargs)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.close()


def recorder(calls: list[str], reply: str = "ok") -> Callable[[str], str]:
    """A handler that records each command and answers ``reply``."""

    def handler(command: str) -> str:
        calls.append(command)
        return reply

    return handler


def test_commands_match_the_controller() -> None:
    assert frozenset(CONTROLLER_COMMANDS) == ipc.COMMANDS


def test_round_trip(server_factory: Callable[..., ipc.InstanceServer]) -> None:
    received: list[str] = []

    def handler(command: str) -> str:
        received.append(command)
        return '{"state":"tracking"}' if command == "status" else "ok"

    server = server_factory(handler)
    assert server.listen()
    assert server.is_listening
    assert ipc.is_running()
    assert ipc.send_command("status") == '{"state":"tracking"}'
    # Commands are normalised (case, surrounding blanks) before they are sent.
    assert ipc.send_command("  Pause ") == "ok"
    assert received == ["status", "pause"]


def test_no_instance(qapp: Any, app_dirs: Path) -> None:
    assert not ipc.is_running()
    assert ipc.send_command("status", timeout_ms=200) is None


def test_invalid_commands_are_rejected_locally(qapp: Any, app_dirs: Path) -> None:
    for bad in ("", "   ", "pause\nquit", "a\rb"):
        with pytest.raises(ValueError, match="invalid command"):
            ipc.send_command(bad)


def test_unknown_command_is_refused_by_the_server(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    handled: list[str] = []
    server = server_factory(recorder(handled))
    assert server.listen()
    reply = ipc.send_command("format-disk")
    assert reply is not None
    assert reply.startswith("error: unknown command 'format-disk'")
    assert "privacy-toggle" in reply  # lists the valid commands
    assert handled == []


def test_handler_errors_become_error_replies(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    def handler(command: str) -> str:
        raise RuntimeError("boom")

    server = server_factory(handler)
    assert server.listen()
    assert ipc.send_command("pause") == "error: boom"


def test_multi_line_replies_are_flattened(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    server = server_factory(lambda c: "first\nsecond")
    assert server.listen()
    assert ipc.send_command("status") == "first second"


def test_slot_can_reply_asynchronously(server_factory: Callable[..., ipc.InstanceServer]) -> None:
    server = server_factory(None)
    seen: list[str] = []

    def on_command(command: str, reply: Callable[[str], None]) -> None:
        seen.append(command)
        QTimer.singleShot(20, lambda: reply(f"later:{command}"))

    server.command_received.connect(on_command)
    assert server.listen()
    assert ipc.send_command("toggle") == "later:toggle"
    assert seen == ["toggle"]


def test_slot_reply_takes_precedence_over_handler(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    handled: list[str] = []
    server = server_factory(recorder(handled, "from handler"))
    server.command_received.connect(lambda command, reply: reply("from slot"))
    assert server.listen()
    assert ipc.send_command("show") == "from slot"
    assert handled == []


def test_second_server_detects_the_first(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    first = server_factory(lambda c: "ok")
    assert first.listen()
    assert first.listen()  # idempotent
    second = server_factory(lambda c: "ok")
    assert not second.listen()
    assert second.another_instance_running
    assert not second.is_listening


def test_close_stops_listening_and_is_idempotent(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    server = server_factory(lambda c: "ok")
    assert server.listen()
    server.close()
    server.close()
    assert not server.is_listening
    assert not ipc.is_running()
    assert ipc.send_command("status", timeout_ms=200) is None
    # The name is free again for a new instance.
    again = server_factory(lambda c: "ok")
    assert again.listen()
    assert ipc.send_command("status") == "ok"


def test_idle_client_is_disconnected(
    server_factory: Callable[..., ipc.InstanceServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "CLIENT_IDLE_TIMEOUT_MS", 50)
    server = server_factory(lambda c: "ok")
    assert server.listen()
    # Connect and send nothing: the server gives up after the idle timeout.
    connected, data = ipc._exchange(server.name, b"", 3000)
    assert connected
    assert data.decode().strip() == "error: no complete request received"


def test_overlong_request_is_refused(server_factory: Callable[..., ipc.InstanceServer]) -> None:
    server = server_factory(lambda c: "ok")
    assert server.listen()
    connected, data = ipc._exchange(server.name, b"x" * (ipc.MAX_REQUEST_BYTES + 10), 3000)
    assert connected
    assert data.decode().strip() == "error: request too long"


def test_invalid_utf8_is_refused(server_factory: Callable[..., ipc.InstanceServer]) -> None:
    server = server_factory(lambda c: "ok")
    assert server.listen()
    connected, data = ipc._exchange(server.name, b"\xff\xfe\n", 3000)
    assert connected
    assert data.decode().strip() == "error: the request is not valid UTF-8"


def test_explicit_server_name(server_factory: Callable[..., ipc.InstanceServer]) -> None:
    name = ipc.server_name() + "-explicit"
    server = server_factory(lambda c: "ok", name=name)
    assert server.name == name
    assert server.listen()
    connected, data = ipc._exchange(name, b"status\n", 3000)
    assert connected
    assert data == b"ok\n"
    # The default name is still free.
    assert not ipc.is_running()


def test_server_name_depends_on_the_config_dir(qapp: Any, tmp_path: Path) -> None:
    from eye_tracker import paths

    try:
        paths.set_base_override(tmp_path / "a")
        first = ipc.server_name()
        paths.set_base_override(tmp_path / "b")
        second = ipc.server_name()
    finally:
        paths.set_base_override(None)
    assert first != second
    assert first.startswith("eye-tracker-")


def test_server_name_uses_the_runtime_dir_on_linux(
    app_dirs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "run"
    runtime.mkdir()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setattr(ipc, "_private_temp_dir", lambda: None)
    full = os.path.join(str(runtime), ipc.paths.ipc_name())
    # The temporary directory's own length must not decide the outcome.
    monkeypatch.setattr(ipc, "_MAX_SOCKET_PATH", len(full.encode()))
    assert ipc.server_name() == full

    # Too long for a Unix socket path, missing or relative: Qt's default location.
    monkeypatch.setattr(ipc, "_MAX_SOCKET_PATH", len(full.encode()) - 1)
    assert ipc.server_name() == ipc.paths.ipc_name()
    monkeypatch.setattr(ipc, "_MAX_SOCKET_PATH", 10_000)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "relative/dir")
    assert ipc.server_name() == ipc.paths.ipc_name()
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert ipc.server_name() == ipc.paths.ipc_name()


def test_server_name_without_runtime_dir_uses_a_private_temp_dir(
    app_dirs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never Qt's shared /tmp, where another account could create the socket first."""
    private = tmp_path / "private"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(ipc, "_private_temp_dir", lambda: str(private))
    monkeypatch.setattr(ipc, "_MAX_SOCKET_PATH", 10_000)  # tmp_path may be long
    assert ipc.server_name() == os.path.join(str(private), ipc.paths.ipc_name())


posix_only = pytest.mark.skipif(not hasattr(os, "getuid"), reason="needs POSIX users")


@posix_only
def test_private_temp_dir_is_created_for_this_user_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc.tempfile, "tempdir", str(tmp_path))
    path = ipc._private_temp_dir()
    assert path == os.path.join(str(tmp_path), f"eye-tracker-{os.getuid()}")
    assert os.stat(path).st_mode & 0o777 == 0o700
    assert ipc._private_temp_dir() == path  # reused


@posix_only
def test_private_temp_dir_refuses_a_directory_others_can_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc.tempfile, "tempdir", str(tmp_path))
    shared = tmp_path / f"eye-tracker-{os.getuid()}"
    shared.mkdir()
    shared.chmod(0o777)
    assert ipc._private_temp_dir() is None
    shared.rmdir()
    shared.symlink_to(tmp_path)  # a symlink planted by someone else
    assert ipc._private_temp_dir() is None


def test_private_temp_dir_needs_posix_users(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "getuid", raising=False)
    assert ipc._private_temp_dir() is None


def test_server_name_ignores_the_runtime_dir_elsewhere(
    app_dirs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert ipc.server_name() == ipc.paths.ipc_name()


def test_send_command_hands_over_the_foreground_right(
    qapp: Any, app_dirs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every request lets the running instance raise its window (Windows focus rules)."""
    calls: list[int] = []
    monkeypatch.setattr(ipc, "_allow_foreground_handoff", lambda: calls.append(1))
    assert ipc.send_command("status", timeout_ms=200) is None  # nothing is listening
    assert calls == [1]


# ------------------------------------------------------------------ instance lock
def test_instance_lock_is_exclusive(app_dirs: Path) -> None:
    first, second = ipc.InstanceLock(), ipc.InstanceLock()
    # In the config directory, or next to the socket when that is a path (Linux).
    assert first.path.name == f"{os.path.basename(ipc.server_name())}.lock"
    assert first.path.parent in (app_dirs, Path(ipc.server_name()).parent)
    assert first.acquire()
    assert first.acquire()  # idempotent
    assert first.is_held
    assert first.enforced
    assert not second.acquire()
    assert not second.is_held
    first.release()
    first.release()  # idempotent
    assert not first.is_held
    assert second.acquire()
    second.release()
    assert first.path.exists()  # kept: deleting it would race with other launches


def test_instance_lock_next_to_a_socket_path(tmp_path: Path) -> None:
    socket_path = str(tmp_path / "run" / "eye-tracker-abc")
    lock = ipc.InstanceLock(socket_path)
    assert lock.path == Path(socket_path + ".lock")
    assert lock.acquire()
    assert lock.path.is_file()
    lock.release()


def test_instance_lock_on_macos_lives_next_to_the_socket(
    app_dirs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In $TMPDIR (local, per user), not in a home directory that may be on a network share."""
    temp = tmp_path / "T"
    temp.mkdir()
    monkeypatch.setattr(ipc.tempfile, "tempdir", str(temp))
    monkeypatch.setattr(sys, "platform", "darwin")
    lock = ipc.InstanceLock("eye-tracker-abc")
    assert lock.path == temp / "eye-tracker-abc.lock"
    monkeypatch.setattr(sys, "platform", "win32")
    assert lock.path == app_dirs / "eye-tracker-abc.lock"


def test_instance_lock_without_file_locking_is_not_enforced(
    app_dirs: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(ipc, "_try_lock", _locking_unsupported)
    lock = ipc.InstanceLock()
    with caplog.at_level(logging.WARNING, logger="eye_tracker.ipc"):
        assert lock.acquire()
    assert not lock.enforced
    assert not lock.is_held
    assert "relying on the socket alone" in caplog.text


def test_instance_lock_that_cannot_be_created_is_not_enforced(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    lock = ipc.InstanceLock(str(blocker / "sock"))  # its "directory" is a file
    assert lock.acquire()
    assert not lock.enforced
    assert not lock.is_held


_LOCK_HOLDER = """
import sys
from pathlib import Path
from eye_tracker import paths
paths.set_base_override(Path(sys.argv[1]))
from eye_tracker import ipc
lock = ipc.InstanceLock()
print("held" if lock.acquire() else "busy", flush=True)
sys.stdin.read()
"""


def test_instance_lock_between_processes_and_after_a_crash(app_dirs: Path) -> None:
    """Of two processes only one gets the lock; a killed holder never leaves it stale."""
    holder = subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER, str(app_dirs)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    lock = ipc.InstanceLock()
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        assert not lock.acquire()
    finally:
        holder.kill()
        holder.wait(10)
        if holder.stdin is not None:
            holder.stdin.close()
        if holder.stdout is not None:
            holder.stdout.close()
    # The OS releases the lock of a dead process (on Windows possibly a moment later).
    deadline = time.monotonic() + 5.0
    while not lock.acquire() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert lock.is_held
    lock.release()


# ----------------------------------------------------------- server and the lock
def test_lock_holder_that_is_not_listening_yet_blocks_a_second_server(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    """The race of two launches: probing the socket alone would let both listen."""
    starting = ipc.InstanceLock()
    assert starting.acquire()
    try:
        assert not ipc.is_running()  # nobody answers yet...
        server = server_factory(lambda c: "ok")
        assert not server.listen()  # ...but the lock decides
        assert server.another_instance_running
        assert not server.is_listening
    finally:
        starting.release()


def test_server_uses_a_lock_taken_at_startup(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    lock = ipc.InstanceLock()
    assert lock.acquire()
    server = server_factory(lambda c: "ok", lock=lock)
    assert server.lock is lock
    assert server.listen()
    assert ipc.send_command("status") == "ok"
    server.close()
    assert not lock.is_held
    assert not ipc.is_running()


def test_server_takes_the_lock_itself_and_releases_it(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    server = server_factory(lambda c: "ok")
    assert server.lock is None
    assert server.listen()
    lock = server.lock
    assert lock is not None
    assert lock.is_held
    server.close()
    assert not lock.is_held


def test_without_file_locks_the_socket_probe_decides(
    server_factory: Callable[..., ipc.InstanceServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "_try_lock", _locking_unsupported)
    first = server_factory(lambda c: "ok")
    assert first.listen()
    second = server_factory(lambda c: "ok")
    assert not second.listen()
    assert second.another_instance_running


def _locking_unsupported(fd: int) -> bool:
    raise OSError(errno.ENOLCK, "No locks available")


# ------------------------------------------------------- waiting for an instance
def test_send_command_waits_for_an_instance_that_is_starting(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    server = server_factory(lambda c: f"hello {c}")
    QTimer.singleShot(150, server.listen)
    started = time.monotonic()
    assert ipc.send_command("show", wait_ms=5000) == "hello show"
    assert time.monotonic() - started >= 0.1


def test_send_command_gives_up_after_the_wait(qapp: Any, app_dirs: Path) -> None:
    started = time.monotonic()
    assert ipc.send_command("show", wait_ms=300) is None
    assert 0.25 <= time.monotonic() - started < 3.0


def test_send_command_does_not_resend_to_a_silent_server(
    server_factory: Callable[..., ipc.InstanceServer],
) -> None:
    server = server_factory(None)
    seen: list[str] = []
    server.command_received.connect(lambda command, reply: seen.append(command))  # never replies
    assert server.listen()
    assert ipc.send_command("show", timeout_ms=200, wait_ms=2000) is None
    assert seen == ["show"]


# ----------------------------------------------------------------- server identity
def test_server_of_another_account_is_ignored(
    server_factory: Callable[..., ipc.InstanceServer],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A squatter on our predictable name gets no request and is not 'already running'."""
    handled: list[str] = []
    impostor = server_factory(recorder(handled))
    assert impostor.listen()
    monkeypatch.setattr(ipc, "_server_is_trusted", lambda socket: False)
    with caplog.at_level(logging.WARNING, logger="eye_tracker.ipc"):
        assert ipc.send_command("show", timeout_ms=300) is None
        assert not ipc.is_running()
    assert "another user account" in caplog.text
    for _ in range(3):
        QCoreApplication.processEvents()
    assert handled == []


def test_trust_check_accepts_our_own_server(
    server_factory: Callable[..., ipc.InstanceServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    verdicts: list[bool] = []
    real = ipc._server_is_trusted

    def spy(socket: Any) -> bool:
        verdicts.append(real(socket))
        return verdicts[-1]

    monkeypatch.setattr(ipc, "_server_is_trusted", spy)
    server = server_factory(lambda c: "ok")
    assert server.listen()
    assert ipc.send_command("status") == "ok"
    assert verdicts == [True]


def test_trust_check_fails_open_when_the_owner_is_unknown() -> None:
    class Broken:
        def socketDescriptor(self) -> int:
            raise RuntimeError("no descriptor")

        def fullServerName(self) -> str:
            raise RuntimeError("no name")

    assert ipc._server_is_trusted(Broken()) is True  # type: ignore[arg-type]


def test_socket_file_owner_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    socket_file = tmp_path / "sock"
    socket_file.write_text("")
    owner = os.stat(socket_file).st_uid
    monkeypatch.setattr(os, "getuid", lambda: owner, raising=False)
    assert ipc._socket_file_is_current_user(str(socket_file)) is True
    monkeypatch.setattr(os, "getuid", lambda: owner + 1, raising=False)
    assert ipc._socket_file_is_current_user(str(socket_file)) is False
    assert ipc._socket_file_is_current_user(str(tmp_path / "missing")) is None
    assert ipc._socket_file_is_current_user("") is None
    monkeypatch.delattr(os, "getuid")
    assert ipc._socket_file_is_current_user(str(socket_file)) is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows token checks")
def test_windows_process_owner_check() -> None:
    assert ipc._process_is_current_user(os.getpid()) is True
    assert ipc._process_is_current_user(4) is False  # the System process
    assert ipc._pipe_server_is_current_user(0) is None  # not a pipe handle


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows fallbacks")
def test_windows_checks_are_inert_elsewhere() -> None:
    assert ipc._pipe_server_is_current_user(5) is None
    assert ipc._process_is_current_user(1) is None
    assert ipc._win_security() is None
