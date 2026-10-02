"""Tests for update/release.py: versions and what is trusted from a release answer."""

from __future__ import annotations

from typing import Any

import pytest

from eye_tracker import REPO_URL
from eye_tracker.update import release as r

TAG = "v0.2.1"
SETUP = "EyeTracker-0.2.1-windows-x64-setup.exe"


def payload(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tag_name": TAG,
        "draft": False,
        "prerelease": False,
        "html_url": "https://evil.example/release",
        "assets": [
            {
                "name": SETUP,
                "size": 63_308_253,
                "browser_download_url": f"{REPO_URL}/releases/download/{TAG}/{SETUP}",
            },
            {
                "name": "SHA256SUMS.txt",
                "size": 520,
                "browser_download_url": f"{REPO_URL}/releases/download/{TAG}/SHA256SUMS.txt",
            },
        ],
    }
    data.update(changes)
    return data


# --------------------------------------------------------------------------- versions
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.2.1", (0, 2, 1)),
        ("v1.10.0", (1, 10, 0)),
        (" 2.0.0 ", (2, 0, 0)),
        ("0.2", None),
        ("0.2.1rc1", None),
        ("0.2.1.dev3", None),
        ("v0.2.1-beta", None),
        ("latest", None),
        ("", None),
        (None, None),
        (7, None),
    ],
)
def test_parse_stable_accepts_only_plain_versions(text: object, expected: object) -> None:
    assert r.parse_stable(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0.2.1", (0, 2, 1)), ("0.3.0.dev1", (0, 3, 0)), ("v1.0.0rc2", (1, 0, 0)), ("dev", None)],
)
def test_the_running_version_keeps_its_leading_numbers(text: str, expected: object) -> None:
    assert r.parse_running(text) == expected


def test_versions_compare_numerically() -> None:
    newer = r.parse_release(payload(tag_name="v0.10.0"))
    assert newer is not None
    assert r.is_newer(newer, "0.9.9")  # not as text
    assert not r.is_newer(newer, "0.10.0")
    assert not r.is_newer(newer, "0.10.1")
    assert r.is_newer(newer, "0.10.0.dev1") is False  # a dev build of it is not older


def test_an_unknown_running_version_is_never_updated() -> None:
    release = r.parse_release(payload())
    assert release is not None
    assert not r.is_newer(release, "unknown")


# ------------------------------------------------------------------------ the release
def test_a_release_keeps_its_installer_and_checksums() -> None:
    release = r.parse_release(payload())
    assert release is not None
    assert release.version == "0.2.1"
    assert release.tag == TAG
    assert release.installer == r.Asset(
        SETUP, f"{REPO_URL}/releases/download/{TAG}/{SETUP}", 63_308_253
    )
    assert release.checksums is not None
    assert release.checksums.name == r.CHECKSUMS_NAME
    assert release.installable


def test_the_page_is_built_here_not_taken_from_the_answer() -> None:
    release = r.parse_release(payload())
    assert release is not None
    assert release.page_url == f"{REPO_URL}/releases/tag/{TAG}"


@pytest.mark.parametrize(
    "bad",
    [
        {"draft": True},
        {"prerelease": True},
        {"tag_name": "v0.3.0-rc1"},
        {"tag_name": "nightly"},
        {"tag_name": None},
        {"tag_name": "v1.2.3\nHost: evil"},
    ],
)
def test_only_stable_published_releases_count(bad: dict[str, Any]) -> None:
    assert r.parse_release(payload(**bad)) is None


@pytest.mark.parametrize("junk", [None, [], "release", 3, {}])
def test_a_junk_answer_is_not_a_release(junk: object) -> None:
    assert r.parse_release(junk) is None


def test_an_asset_must_be_served_from_the_repository() -> None:
    data = payload()
    data["assets"][0]["browser_download_url"] = f"https://evil.example/{SETUP}"
    release = r.parse_release(data)
    assert release is not None
    assert release.installer is None
    assert not release.installable


def test_an_asset_of_another_tag_is_refused() -> None:
    data = payload()
    data["assets"][0]["browser_download_url"] = f"{REPO_URL}/releases/download/v9.9.9/{SETUP}"
    release = r.parse_release(data)
    assert release is not None
    assert release.installer is None


@pytest.mark.parametrize("size", [0, -1, True, "63", None, r.MAX_INSTALLER_BYTES + 1])
def test_an_asset_needs_a_sane_size(size: object) -> None:
    data = payload()
    data["assets"][0]["size"] = size
    release = r.parse_release(data)
    assert release is not None
    assert release.installer is None


def test_without_a_checksum_list_nothing_is_installable() -> None:
    data = payload()
    data["assets"] = data["assets"][:1]
    release = r.parse_release(data)
    assert release is not None
    assert release.installer is not None
    assert release.checksums is None
    assert not release.installable


def test_a_release_without_assets_still_names_its_version() -> None:
    release = r.parse_release(payload(assets=None))
    assert release is not None
    assert release.installer is None
    assert release.version == "0.2.1"


def test_names_and_addresses() -> None:
    assert r.installer_name("0.2.1") == SETUP
    assert r.download_url(TAG, SETUP) == f"{REPO_URL}/releases/download/{TAG}/{SETUP}"
    assert (
        r.LATEST_RELEASE_URL == "https://api.github.com/repos/bugraskl/eye-tracker/releases/latest"
    )


def test_the_names_the_app_looks_for_are_the_ones_the_release_workflow_publishes() -> None:
    from pathlib import Path

    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
    if not workflow.is_file():
        pytest.skip("the workflows are not part of this source tree")
    text = workflow.read_text(encoding="utf-8")
    # Renaming either file would make every installed app stop finding its update.
    assert r.installer_name("$VERSION") in text
    assert r.CHECKSUMS_NAME in text
