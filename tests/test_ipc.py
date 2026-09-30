"""Tests for eye_tracker.ipc: the single-instance socket and command channel.

Server and client live in the same thread: the client runs a local event loop
while it waits, which also serves the server. Every test uses the ``app_dirs``
fixture, so the socket name (derived from the config directory) is unique.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QTimer

from eye_tracker import ipc
from eye_tracker.engine.controller import COMMANDS as CONTROLLER_COMMANDS


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


def test_server_name_ignores_the_runtime_dir_elsewhere(
    app_dirs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert ipc.server_name() == ipc.paths.ipc_name()
