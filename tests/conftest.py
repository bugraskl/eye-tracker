"""Shared pytest fixtures.

Qt runs on the ``offscreen`` platform so the suite works on headless CI runners.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@pytest.fixture(scope="session")
def qapp():
    """A process-wide QApplication (created once, offscreen)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["eye-tracker-tests"])
    app.setQuitOnLastWindowClosed(False)
    return app


@pytest.fixture
def app_dirs(tmp_path: Path) -> Iterator[Path]:
    """Redirect config, data and logs into a temporary directory."""
    from eye_tracker import paths

    paths.set_base_override(tmp_path)
    try:
        yield tmp_path
    finally:
        paths.set_base_override(None)


@pytest.fixture
def face_image() -> Path:
    """Path to a photo containing one frontal face.

    Face photos are not committed to the repository. Set ``EYE_TRACKER_TEST_FACE``
    to an image path to enable tests that need a real face; otherwise they skip.
    """
    value = os.environ.get("EYE_TRACKER_TEST_FACE")
    if not value or not Path(value).is_file():
        pytest.skip("set EYE_TRACKER_TEST_FACE to a face photo to run this test")
    return Path(value)
