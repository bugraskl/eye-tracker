"""Tests for update/fetch.py: what the update check may fetch, without a network."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from eye_tracker.update import fetch
from eye_tracker.update.fetch import Cancelled, Fetcher, NetworkError


@dataclass
class FakeResponse:
    status: int = 200
    body: bytes = b""
    location: str | None = None
    announced: int | None = -1  # -1: the real length of the body
    closed: bool = False
    _pos: int = 0

    @property
    def content_length(self) -> int | None:
        return len(self.body) if self.announced == -1 else self.announced

    def read(self, size: int) -> bytes:
        chunk = self.body[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeTransport:
    """Answers by (host, path); records what it was asked."""

    answers: dict[tuple[str, str], FakeResponse] = field(default_factory=dict)
    asked: list[tuple[str, str, str | None]] = field(default_factory=list)
    closed: bool = False

    def open(self, host: str, path: str, accept: str | None) -> FakeResponse:
        self.asked.append((host, path, accept))
        return self.answers[(host, path)]

    def close(self) -> None:
        self.closed = True


def redirect(location: str, status: int = 302) -> FakeResponse:
    return FakeResponse(status=status, location=location)


# ----------------------------------------------------------------------------- addresses
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://api.github.com/repos/a/b/releases/latest",
            ("api.github.com", "/repos/a/b/releases/latest"),
        ),
        ("https://GitHub.com/a/b?x=1", ("github.com", "/a/b?x=1")),
        ("https://github.com", ("github.com", "/")),
        (
            "https://release-assets.githubusercontent.com/x/y?sig=abc%3D",
            ("release-assets.githubusercontent.com", "/x/y?sig=abc%3D"),
        ),
    ],
)
def test_split_url_accepts_github_hosts(url: str, expected: tuple[str, str]) -> None:
    assert fetch.split_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/a",  # not https
        "https://evil.example/a",
        "https://github.com.evil.example/a",
        "https://evil.example/github.com",
        "https://user:pw@github.com/a",  # credentials
        "https://github.com:8443/a",  # port
        "https://github.com@evil.example/a",
        "//github.com/a",
        "ftp://github.com/a",
        "https://github.com/a b",
        "https://github.com/a#frag",
        "",
        "github.com/a",
    ],
)
def test_split_url_refuses_everything_else(url: str) -> None:
    with pytest.raises(NetworkError):
        fetch.split_url(url)


def test_only_github_hosts_are_allowed() -> None:
    assert {
        "api.github.com",
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    } == fetch.ALLOWED_HOSTS


def test_a_location_path_stays_on_the_host() -> None:
    assert fetch.resolve_location("github.com", "/a/b") == "https://github.com/a/b"
    assert fetch.resolve_location("github.com", "https://x.test/a") == "https://x.test/a"
    # A protocol-relative address is not a path of this host.
    assert fetch.resolve_location("github.com", "//evil.example/a") == "//evil.example/a"


# ------------------------------------------------------------------------------- get
def test_get_returns_the_body_and_sends_only_the_accept_header() -> None:
    transport = FakeTransport({("api.github.com", "/x"): FakeResponse(body=b"hello")})
    with Fetcher(transport) as fetcher:
        assert fetcher.get("https://api.github.com/x", max_bytes=100, accept="a/b") == b"hello"
    assert transport.asked == [("api.github.com", "/x", "a/b")]
    assert not transport.closed  # a transport that was passed in is not ours to close


def test_get_follows_redirects_through_allowed_hosts() -> None:
    final = FakeResponse(body=b"file")
    transport = FakeTransport(
        {
            ("github.com", "/a"): redirect("https://release-assets.githubusercontent.com/b?s=1"),
            ("release-assets.githubusercontent.com", "/b?s=1"): final,
        }
    )
    assert Fetcher(transport).get("https://github.com/a", max_bytes=10) == b"file"
    assert final.closed


def test_a_relative_redirect_stays_on_the_host() -> None:
    transport = FakeTransport(
        {("github.com", "/a"): redirect("/b"), ("github.com", "/b"): FakeResponse(body=b"x")}
    )
    assert Fetcher(transport).get("https://github.com/a", max_bytes=10) == b"x"


def test_a_redirect_to_another_host_is_refused_without_asking_it() -> None:
    first = redirect("https://evil.example/file")
    transport = FakeTransport({("github.com", "/a"): first})
    with pytest.raises(NetworkError, match="not a GitHub host"):
        Fetcher(transport).get("https://github.com/a", max_bytes=10)
    assert [host for host, _, _ in transport.asked] == ["github.com"]
    assert first.closed


@pytest.mark.parametrize("location", ["http://github.com/b", "ftp://github.com/b", "//evil/b"])
def test_a_redirect_that_leaves_https_is_refused(location: str) -> None:
    transport = FakeTransport({("github.com", "/a"): redirect(location)})
    with pytest.raises(NetworkError):
        Fetcher(transport).get("https://github.com/a", max_bytes=10)


def test_redirect_loops_end() -> None:
    transport = FakeTransport({("github.com", "/a"): redirect("https://github.com/a")})
    with pytest.raises(NetworkError, match="Too many redirects"):
        Fetcher(transport).get("https://github.com/a", max_bytes=10)
    assert len(transport.asked) == fetch.MAX_REDIRECTS + 1


@pytest.mark.parametrize("status", [403, 404, 500, 301, 304])
def test_other_statuses_are_errors(status: int) -> None:
    transport = FakeTransport({("github.com", "/a"): FakeResponse(status=status)})
    with pytest.raises(NetworkError, match=f"HTTP {status}"):
        Fetcher(transport).get("https://github.com/a", max_bytes=10)


def test_an_announced_length_over_the_limit_is_refused_before_reading() -> None:
    response = FakeResponse(body=b"x", announced=11)
    transport = FakeTransport({("github.com", "/a"): response})
    with pytest.raises(NetworkError, match="larger"):
        Fetcher(transport).get("https://github.com/a", max_bytes=10)
    assert response.closed
    assert response._pos == 0


def test_a_body_over_the_limit_is_refused_even_if_it_lied_about_its_length() -> None:
    response = FakeResponse(body=b"x" * 50, announced=None)
    transport = FakeTransport({("github.com", "/a"): response})
    with pytest.raises(NetworkError, match="larger"):
        Fetcher(transport).get("https://github.com/a", max_bytes=10)
    assert response.closed


def test_get_reads_a_body_in_several_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "CHUNK_BYTES", 4)
    transport = FakeTransport({("github.com", "/a"): FakeResponse(body=b"0123456789")})
    assert Fetcher(transport).get("https://github.com/a", max_bytes=10) == b"0123456789"


def test_a_fetcher_closes_the_transport_it_made(monkeypatch: pytest.MonkeyPatch) -> None:
    made = FakeTransport({("github.com", "/a"): FakeResponse(body=b"x")})
    monkeypatch.setattr(fetch, "default_transport", lambda: made)
    with Fetcher() as fetcher:
        fetcher.get("https://github.com/a", max_bytes=10)
    assert made.closed


def test_without_a_transport_other_platforms_say_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "supported", lambda: False)
    with pytest.raises(NetworkError, match="not available"):
        fetch.default_transport()


# -------------------------------------------------------------------------- download
def test_download_writes_the_file_and_reports_progress(tmp_path: Path) -> None:
    transport = FakeTransport({("github.com", "/f"): FakeResponse(body=b"x" * 100)})
    seen: list[tuple[int, int]] = []
    out = tmp_path / "f.part"
    size = Fetcher(transport).download(
        "https://github.com/f", out, max_bytes=100, progress=lambda d, t: seen.append((d, t))
    )
    assert size == 100
    assert out.read_bytes() == b"x" * 100
    assert seen[-1] == (100, 100)


def test_download_stops_at_the_limit(tmp_path: Path) -> None:
    response = FakeResponse(body=b"x" * 100, announced=None)
    transport = FakeTransport({("github.com", "/f"): response})
    with pytest.raises(NetworkError, match="larger"):
        Fetcher(transport).download("https://github.com/f", tmp_path / "f", max_bytes=50)
    assert response.closed


def test_download_can_be_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "CHUNK_BYTES", 10)
    response = FakeResponse(body=b"x" * 100)
    transport = FakeTransport({("github.com", "/f"): response})
    chunks = iter([False, False, True])
    with pytest.raises(Cancelled):
        Fetcher(transport).download(
            "https://github.com/f", tmp_path / "f", max_bytes=100, cancelled=lambda: next(chunks)
        )
    assert response.closed
    assert response._pos == 20
