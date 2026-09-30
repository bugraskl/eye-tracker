"""Tests for scripts/fetch_models.py's archive handling (the MediaPipe ``.task`` bundle).

The single-file download path is covered in ``test_build_scripts.py``. Nothing
here touches the network: ``urlopen`` is replaced by a fake.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    name = "fetch_models_archive_under_test"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "fetch_models.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


fetch_models = _load()

MEMBERS = {"weights.tflite": b"landmark weights", "geometry.binarypb": b"canonical face"}


def _zip(members: dict[str, bytes], extra: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as bundle:
        for name, data in {**members, **(extra or {})}.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def _bundle_model(archive: bytes, members: dict[str, bytes] = MEMBERS) -> Any:
    return fetch_models.Model(
        filename="bundle.task",
        url="https://example.invalid/bundle.task",
        sha256=hashlib.sha256(archive).hexdigest(),
        license="Apache-2.0",
        homepage="https://example.invalid",
        description="test bundle",
        members=tuple(
            fetch_models.Member(name, hashlib.sha256(data).hexdigest())
            for name, data in members.items()
        ),
    )


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Bodies returned by successive (fake) downloads."""
    bodies: list[bytes] = []

    def urlopen(request: Any, timeout: float) -> _FakeResponse:
        return _FakeResponse(bodies.pop(0))

    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(fetch_models.time, "sleep", lambda _s: None)
    return bodies


def test_installed_files_of_a_bundle() -> None:
    model = _bundle_model(_zip(MEMBERS))
    assert [name for name, _ in model.installed()] == list(MEMBERS)


def test_bundle_download_installs_only_the_verified_members(
    serve: list[bytes], tmp_path: Path
) -> None:
    archive = _zip(MEMBERS, extra={"unused.tflite": b"not needed"})
    model = _bundle_model(archive)
    serve.append(archive)
    first = fetch_models.download(model, tmp_path)
    assert first == tmp_path / "weights.tflite"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(MEMBERS)
    for name, data in MEMBERS.items():
        assert (tmp_path / name).read_bytes() == data
    assert fetch_models.verify(model, tmp_path) == "ok"


def test_member_checksum_mismatch_installs_nothing(serve: list[bytes], tmp_path: Path) -> None:
    archive = _zip({"weights.tflite": b"tampered", "geometry.binarypb": b"canonical face"})
    model = _bundle_model(archive, members=MEMBERS)  # pins the genuine members
    serve.append(archive)
    with pytest.raises(fetch_models.DownloadError, match=r"checksum mismatch for weights\.tflite"):
        fetch_models.download(model, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_missing_member(serve: list[bytes], tmp_path: Path) -> None:
    archive = _zip({"weights.tflite": MEMBERS["weights.tflite"]})
    model = _bundle_model(archive, members=MEMBERS)
    serve.append(archive)
    with pytest.raises(fetch_models.DownloadError, match=r"no member 'geometry\.binarypb'"):
        fetch_models.download(model, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_not_a_zip(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.task"
    archive.write_bytes(b"definitely not a zip file")
    model = _bundle_model(archive.read_bytes())
    dest = tmp_path / "models"
    dest.mkdir()
    with pytest.raises(fetch_models.DownloadError, match="not a readable zip archive"):
        fetch_models.extract_members(model, archive, dest)
    assert list(dest.iterdir()) == []


def test_member_paths_never_escape_the_model_directory(tmp_path: Path) -> None:
    members = {"../outside.bin": b"payload"}
    archive_bytes = _zip(members)
    archive = tmp_path / "bundle.task"
    archive.write_bytes(archive_bytes)
    dest = tmp_path / "models"
    dest.mkdir()
    installed = fetch_models.extract_members(_bundle_model(archive_bytes, members), archive, dest)
    assert installed == [dest / "outside.bin"]
    assert not (tmp_path / "outside.bin").exists()


def test_real_bundle_pins_match_the_committed_models() -> None:
    (task,) = [m for m in fetch_models.MODELS if m.members]
    assert task.filename == "face_landmarker.task"
    assert {m.name for m in task.members} == {
        "face_landmarks_detector.tflite",
        "geometry_pipeline_metadata_landmarks.binarypb",
    }
    assert fetch_models.verify(task, fetch_models.DEFAULT_DEST) == "ok"
