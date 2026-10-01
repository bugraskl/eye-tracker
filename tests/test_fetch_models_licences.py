"""The bundled models' licence texts ship with them and are verified (r2-vision-07).

Apache-2.0 requires giving recipients a copy of the licence and MIT requires its
copyright and permission notice in all copies, so ``NOTICE.md`` linking to them
is not enough. Nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from eye_tracker import paths
from eye_tracker.vision.backends import MODEL_FILES

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "fetch_models.py"

if not SCRIPT.is_file():  # an sdist ships the tests but not the build tooling
    pytest.skip("scripts/fetch_models.py is not part of this source tree", allow_module_level=True)


def _load() -> ModuleType:
    name = "fetch_models_licences_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


fetch_models = _load()
MODEL_DIR = paths.model_path("NOTICE.md").parent


def _copy_models(dest: Path, *, licences: bool) -> None:
    """The committed models (and, optionally, their licence texts) in ``dest``."""
    for name in MODEL_FILES:
        shutil.copyfile(MODEL_DIR / name, dest / name)
    if licences:
        shutil.copytree(MODEL_DIR / "licenses", dest / "licenses")


def test_licence_texts_are_committed_next_to_the_models() -> None:
    for text in fetch_models.LICENCE_TEXTS:
        path = MODEL_DIR / text.path
        assert path.is_file(), text.path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == text.sha256, text.path
        assert fetch_models.verify_licence(text, MODEL_DIR) == "ok"


def test_every_model_file_is_covered_by_exactly_one_licence_text() -> None:
    covered = [name for text in fetch_models.LICENCE_TEXTS for name in text.covers]
    assert sorted(covered) == sorted(MODEL_FILES)


def test_licence_texts_say_what_they_must() -> None:
    by_name = {
        # Bytes, not read_text(): universal newlines would hide a CRLF.
        Path(t.path).name: (MODEL_DIR / t.path).read_bytes().decode("utf-8")
        for t in fetch_models.LICENCE_TEXTS
    }
    apache = by_name["LICENSE-APACHE-2.0.txt"]
    assert "Apache License" in apache
    assert "Version 2.0, January 2004" in apache
    assert "END OF TERMS AND CONDITIONS" in apache
    yunet = by_name["LICENSE-YUNET.txt"]
    assert yunet.startswith("MIT License\n")
    assert "Copyright (c) 2020 Shiqi Yu" in yunet
    assert "The above copyright notice and this permission notice shall be included" in yunet
    for text in by_name.values():
        assert "\r" not in text  # LF everywhere, so the pinned checksums hold on every OS


def test_notice_references_every_licence_text() -> None:
    notice = (MODEL_DIR / "NOTICE.md").read_text("utf-8")
    for text in fetch_models.LICENCE_TEXTS:
        assert f"]({text.path})" in notice, text.path  # a working relative link
        assert text.sha256 in notice, text.path


def test_check_passes_with_the_committed_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _copy_models(tmp_path, licences=True)
    assert fetch_models.main(["--check", "--dest", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for text in fetch_models.LICENCE_TEXTS:
        assert f"licence  {text.path}" in out


def test_check_fails_without_the_licence_texts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _copy_models(tmp_path, licences=False)
    assert fetch_models.main(["--check", "--dest", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "missing  licenses/LICENSE-APACHE-2.0.txt" in captured.out
    assert "missing  licenses/LICENSE-YUNET.txt" in captured.out
    assert f"{len(fetch_models.LICENCE_TEXTS)} licence text(s) missing or altered" in captured.err
    assert "model(s) missing" not in captured.err  # the models themselves are fine


def test_check_fails_when_a_licence_text_was_altered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _copy_models(tmp_path, licences=True)
    yunet = tmp_path / "licenses" / "LICENSE-YUNET.txt"
    yunet.write_bytes(yunet.read_bytes().replace(b"Shiqi Yu", b"Somebody Else"))
    assert fetch_models.main(["--check", "--dest", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "altered  licenses/LICENSE-YUNET.txt" in captured.out
    assert "1 licence text(s) missing or altered" in captured.err


def test_download_mode_only_warns_about_licence_texts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fetching models into another folder is fine; the texts come from git, not the web."""
    payload = b"weights"
    model = fetch_models.Model(
        filename="model.bin",
        url="https://example.invalid/model.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        license="MIT",
        homepage="https://example.invalid",
        description="test model",
    )
    (tmp_path / "model.bin").write_bytes(payload)  # present: nothing to download

    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("licence texts are never downloaded")

    monkeypatch.setattr(fetch_models, "MODELS", (model,))
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", no_network)
    assert fetch_models.main(["--dest", str(tmp_path)]) == 0
    assert "warning: 2 licence text(s) missing or altered" in capsys.readouterr().err
