"""Tests for update/service.py: when to look, what to announce, how an install goes."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from eye_tracker import REPO_URL
from eye_tracker.update import fetch, installer
from eye_tracker.update import service as svc
from eye_tracker.update.fetch import Fetcher
from eye_tracker.update.release import download_url
from eye_tracker.update.service import Phase, UpdateService, UpdateState
from test_update_fetch import FakeResponse, FakeTransport

TAG = "v0.2.2"
NAME = "EyeTracker-0.2.2-windows-x64-setup.exe"
BODY = b"MZ" + b"\x01" * 4000
API = ("api.github.com", "/repos/bugraskl/eye-tracker/releases/latest")
BASE = f"/bugraskl/eye-tracker/releases/download/{TAG}"


def release_json(tag: str = TAG) -> bytes:
    name = f"EyeTracker-{tag[1:]}-windows-x64-setup.exe"
    return json.dumps(
        {
            "tag_name": tag,
            "draft": False,
            "prerelease": False,
            "assets": [
                {
                    "name": name,
                    "size": len(BODY),
                    "browser_download_url": download_url(tag, name),
                },
                {
                    "name": "SHA256SUMS.txt",
                    "size": 150,
                    "browser_download_url": download_url(tag, "SHA256SUMS.txt"),
                },
            ],
        }
    ).encode()


class FreshTransport(FakeTransport):
    """Every request gets an unread copy of the answer (a check can run twice)."""

    def open(self, host: str, path: str, accept: str | None) -> FakeResponse:
        template = self.answers[(host, path)]
        self.asked.append((host, path, accept))
        return replace(template, _pos=0, closed=False)


def transport(tag: str = TAG, *, body: bytes = BODY) -> FakeTransport:
    name = f"EyeTracker-{tag[1:]}-windows-x64-setup.exe"
    listing = f"{hashlib.sha256(BODY).hexdigest()}  {name}\n".encode()
    return FreshTransport(
        {
            API: FakeResponse(body=release_json(tag)),
            ("github.com", f"/bugraskl/eye-tracker/releases/download/{tag}/SHA256SUMS.txt"): (
                FakeResponse(body=listing)
            ),
            ("github.com", f"/bugraskl/eye-tracker/releases/download/{tag}/{name}"): FakeResponse(
                body=body
            ),
        }
    )


@pytest.fixture
def make(
    qapp: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., UpdateService]:
    monkeypatch.setattr(svc, "supported", lambda: True)
    monkeypatch.setattr(installer, "updates_dir", lambda: tmp_path / "updates")
    made: list[UpdateService] = []

    def factory(
        fake: FakeTransport | None = None,
        *,
        running: str = "0.2.1",
        kind: str = installer.KIND_SETUP,
        clock: Callable[[], float] = time.time,
        launcher: Callable[[Path], None] | None = None,
    ) -> UpdateService:
        the_transport = fake if fake is not None else transport()
        service = UpdateService(
            fetcher_factory=lambda: Fetcher(the_transport),
            clock=clock,
            running_version=running,
            memory_path=tmp_path / "updates.json",
            kind=kind,
            launcher=launcher or (lambda path: None),
        )
        made.append(service)
        return service

    yield factory
    for service in made:
        service.shutdown()
        service.deleteLater()


def wait_for(qapp: Any, predicate: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        qapp.processEvents()
        time.sleep(0.005)


def phases(service: UpdateService) -> list[Phase]:
    seen: list[Phase] = []
    service.state_changed.connect(lambda state: seen.append(state.phase))
    return seen


# ------------------------------------------------------------------------- checking
def test_a_manual_check_finds_a_newer_release(
    qapp: Any, make: Callable[..., UpdateService]
) -> None:
    service = make()
    seen = phases(service)
    assert service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE)
    assert seen == [Phase.CHECKING, Phase.AVAILABLE]
    assert service.state.release is not None
    assert service.state.release.version == "0.2.2"
    assert service.state.manual
    assert service.can_install


def test_a_manual_check_of_the_latest_version_says_so(
    qapp: Any, make: Callable[..., UpdateService]
) -> None:
    service = make(running="0.2.2")
    service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.UP_TO_DATE)
    assert not service.can_install


def test_a_failed_check_reports_why(qapp: Any, make: Callable[..., UpdateService]) -> None:
    service = make(FakeTransport({API: FakeResponse(status=403)}))
    service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.FAILED)
    assert "403" in service.state.message
    assert service.state.release is None


def test_a_junk_answer_is_a_failed_check(qapp: Any, make: Callable[..., UpdateService]) -> None:
    service = make(FakeTransport({API: FakeResponse(body=b"<html>captive portal</html>")}))
    service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.FAILED)
    assert "could not be read" in service.state.message


def test_checking_is_unavailable_where_there_is_no_transport(
    make: Callable[..., UpdateService], monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make()
    monkeypatch.setattr(svc, "supported", lambda: False)
    assert not service.available
    assert not service.check_now()
    service.set_auto(True)
    assert not service._timer.isActive()


def test_a_second_check_waits_for_the_first(qapp: Any, make: Callable[..., UpdateService]) -> None:
    service = make()
    assert service.check_now()
    assert not service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE)


# ------------------------------------------------------------------- daily schedule
def test_the_daily_check_is_off_unless_asked_for(make: Callable[..., UpdateService]) -> None:
    service = make()
    service.start(auto=False)
    assert not service._timer.isActive()
    service.set_auto(True)
    assert service._timer.isActive()
    service.set_auto(False)
    assert not service._timer.isActive()


def test_the_first_check_is_not_at_startup(make: Callable[..., UpdateService]) -> None:
    service = make()
    service.start(auto=True)
    assert service._timer.remainingTime() > 60_000


def test_an_automatic_check_announces_a_version_once(
    qapp: Any, make: Callable[..., UpdateService], tmp_path: Path
) -> None:
    now = [1_000_000.0]
    service = make(clock=lambda: now[0])
    service.set_auto(True)
    announced: list[str] = []
    service.announce.connect(lambda release: announced.append(release.version))

    service._tick()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE)
    wait_for(qapp, lambda: not service.busy)
    assert announced == ["0.2.2"]

    now[0] += svc.CHECK_INTERVAL_S + 1  # the next day: found again, not announced again
    service._tick()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE and not service.busy)
    qapp.processEvents()
    assert announced == ["0.2.2"]
    assert json.loads((tmp_path / "updates.json").read_text())["notified"] == "0.2.2"


def test_the_check_is_not_repeated_within_a_day(
    qapp: Any, make: Callable[..., UpdateService]
) -> None:
    now = [5_000_000.0]
    fake = transport()
    service = make(fake, clock=lambda: now[0])
    service.set_auto(True)
    service._tick()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE and not service.busy)
    asked = len(fake.asked)
    now[0] += 3600
    service._tick()
    qapp.processEvents()
    assert len(fake.asked) == asked


def test_a_clock_set_back_does_not_block_checks(
    qapp: Any, make: Callable[..., UpdateService], tmp_path: Path
) -> None:
    (tmp_path / "updates.json").write_text(json.dumps({"last_check": 9e12, "notified": ""}))
    service = make(clock=lambda: 1000.0)
    service.set_auto(True)
    service._tick()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE)


def test_a_manual_check_does_not_announce(qapp: Any, make: Callable[..., UpdateService]) -> None:
    service = make()
    announced: list[object] = []
    service.announce.connect(announced.append)
    service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE)
    qapp.processEvents()
    assert announced == []


def test_the_memory_survives_a_bad_file(make: Callable[..., UpdateService], tmp_path: Path) -> None:
    (tmp_path / "updates.json").write_text("{not json")
    service = make()
    assert service._memory.last_check == 0.0
    (tmp_path / "updates.json").write_text(json.dumps({"last_check": True, "notified": 3}))
    assert make()._memory.last_check == 0.0


# --------------------------------------------------------------------------- install
def test_the_installer_is_downloaded_verified_and_started(
    qapp: Any, make: Callable[..., UpdateService], tmp_path: Path
) -> None:
    started: list[Path] = []
    service = make(launcher=started.append)
    seen = phases(service)
    service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE)
    wait_for(qapp, lambda: not service.busy)
    assert service.install()
    wait_for(qapp, lambda: service.state.phase is Phase.INSTALLING)
    assert started == [tmp_path / "updates" / NAME]
    assert started[0].read_bytes() == BODY
    assert Phase.DOWNLOADING in seen
    assert seen[-1] is Phase.INSTALLING


def test_a_tampered_installer_is_not_started(
    qapp: Any, make: Callable[..., UpdateService], tmp_path: Path
) -> None:
    started: list[Path] = []
    service = make(transport(body=BODY[:-1] + b"!"), launcher=started.append)
    service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE and not service.busy)
    service.install()
    wait_for(qapp, lambda: service.state.phase is Phase.FAILED)
    assert "checksum" in service.state.message
    assert started == []
    assert list((tmp_path / "updates").iterdir()) == []
    assert service.can_install  # try again


@pytest.mark.parametrize("kind", [installer.KIND_PORTABLE, installer.KIND_UNSUPPORTED])
def test_copies_the_app_does_not_own_are_never_installed(
    qapp: Any, make: Callable[..., UpdateService], kind: str
) -> None:
    started: list[Path] = []
    service = make(kind=kind, launcher=started.append)
    service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE and not service.busy)
    assert not service.can_install
    assert not service.install()
    assert started == []


def test_nothing_is_installed_before_a_check_found_something(
    make: Callable[..., UpdateService],
) -> None:
    service = make()
    assert not service.install()
    assert service.state == UpdateState()


def test_a_failing_launcher_is_reported(qapp: Any, make: Callable[..., UpdateService]) -> None:
    def refuse(path: Path) -> None:
        raise installer.UpdateError("blocked")

    service = make(launcher=refuse)
    service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE and not service.busy)
    service.install()
    wait_for(qapp, lambda: service.state.phase is Phase.FAILED)
    assert service.state.message == "blocked"


class SlowResponse(FakeResponse):
    """A body that arrives slowly, so that a download can be cancelled half way."""

    def read(self, size: int) -> bytes:
        time.sleep(0.01)
        return super().read(size)


def test_a_download_can_be_cancelled(
    qapp: Any, make: Callable[..., UpdateService], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "CHUNK_BYTES", 100)
    fake = transport()
    installer_path = ("github.com", f"{BASE}/{NAME}")
    fake.answers[installer_path] = SlowResponse(body=BODY)
    started: list[Path] = []
    service = make(fake, launcher=started.append)
    service.check_now()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE and not service.busy)
    assert service.install()
    service.cancel()
    wait_for(qapp, lambda: service.state.phase is Phase.AVAILABLE)
    assert started == []
    assert list((tmp_path / "updates").iterdir()) == []
    assert service.can_install  # the user may start it again


def test_old_downloads_are_cleared_at_start(
    make: Callable[..., UpdateService], tmp_path: Path
) -> None:
    folder = tmp_path / "updates"
    folder.mkdir()
    (folder / NAME).write_bytes(b"left over")
    make().start(auto=False)
    assert list(folder.iterdir()) == []


def test_the_release_page_is_the_repository_s() -> None:
    assert svc.LATEST_RELEASE_URL.startswith("https://api.github.com/repos/")
    assert REPO_URL.endswith("/eye-tracker")
