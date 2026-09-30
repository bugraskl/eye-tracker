"""Persistence of calibrations as a small, human-readable JSON file.

Only numbers are stored: the feature vectors (head pose angles, head position,
iris offsets), the screen points they were labelled with and the fitted model.
Camera frames are never part of a calibration.

The file is pretty-printed with one sample per line so it stays readable and
reasonably small (a two-monitor calibration is roughly 50 KB). Loading never
raises: a missing, unreadable, corrupt or too new file yields ``None`` and the
app asks for a new calibration.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import atomic_write_text
from ..types import Monitor, Rect, layout_signature
from .calibration import CalibrationSample
from .model import GazeModel

log = logging.getLogger(__name__)

CALIBRATION_VERSION = 1

#: Marker that identifies the file type.
FILE_FORMAT = "eye-tracker-calibration"

# Samples are measurements, so six decimals are far below their noise. Model
# parameters keep ten significant digits so a reloaded model predicts the same
# pixels as the one that was saved.
_SAMPLE_DECIMALS = 6
_MODEL_DIGITS = 10


def utc_now_iso() -> str:
    """Current time as ISO 8601 UTC with second precision, e.g. ``2026-09-30T12:00:00+00:00``."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(eq=False)
class CalibrationData:
    """A calibration and everything needed to decide whether it is still valid."""

    backend: str
    feature_version: str
    layout_signature: str
    monitors: list[Monitor]
    samples: list[CalibrationSample]
    implicit_samples: list[CalibrationSample]
    model: GazeModel
    report: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def is_compatible(
        self,
        backend_name: str,
        feature_version: str,
        monitors: Sequence[Monitor],
    ) -> tuple[bool, str]:
        """``(True, "")`` if usable with this backend and monitor layout, else
        ``(False, reason)`` where ``reason`` is a short lower-case phrase such as
        ``"the monitor layout changed"``."""
        if backend_name != self.backend:
            return (
                False,
                f"calibrated with the {self.backend} backend, but {backend_name} is in use",
            )
        if feature_version != self.feature_version:
            return False, (
                f"the {backend_name} backend's measurements changed "
                f"({self.feature_version} → {feature_version})"
            )
        if layout_signature(monitors) != self.layout_signature:
            return False, "the monitor layout changed"
        if not self.model.is_fitted:
            return False, "the calibration has no fitted model"
        return True, ""

    @property
    def grade(self) -> str | None:
        """Quality grade from the report (``"excellent"`` … ``"poor"``), if known."""
        value = self.report.get("grade")
        return value if isinstance(value, str) else None


# ------------------------------------------------------------------------ save
def save_calibration(path: Path, data: CalibrationData) -> None:
    """Write ``data`` to ``path`` atomically. Raises ``OSError`` if writing fails."""
    doc = {
        "format": FILE_FORMAT,
        "version": CALIBRATION_VERSION,
        "created_at": data.created_at,
        "backend": data.backend,
        "feature_version": data.feature_version,
        "layout_signature": data.layout_signature,
        "monitors": [_monitor_to_dict(m) for m in data.monitors],
        "report": _json_safe(data.report),
        "model": _round_model(data.model.to_dict()),
        "samples": _samples_to_list(data.samples),
        "implicit_samples": _samples_to_list(data.implicit_samples),
    }
    atomic_write_text(Path(path), _dumps(doc))
    log.debug(
        "Saved calibration (%d + %d samples) to %s",
        len(data.samples),
        len(data.implicit_samples),
        path,
    )


# ------------------------------------------------------------------------ load
def load_calibration(path: Path) -> CalibrationData | None:
    """Read a calibration; ``None`` if the file is missing, corrupt or unsupported."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.warning("Could not read calibration %s: %s", path, exc)
        return None
    try:
        doc = json.loads(text)
        return _from_doc(doc)
    except _NewerVersion as exc:
        log.warning("Calibration %s was written by a newer Eye Tracker (%s); ignored", path, exc)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        # json.JSONDecodeError is a ValueError.
        log.warning("Ignoring corrupt calibration file %s: %s", path, exc)
    return None


class _NewerVersion(Exception):
    pass


def _from_doc(doc: Any) -> CalibrationData:
    if not isinstance(doc, dict):
        raise ValueError("root is not an object")
    if doc.get("format", FILE_FORMAT) != FILE_FORMAT:
        raise ValueError(f"not a calibration file (format {doc.get('format')!r})")
    version = doc.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"invalid version {version!r}")
    if version > CALIBRATION_VERSION:
        raise _NewerVersion(f"version {version}")

    model = GazeModel.from_dict(doc["model"])
    samples = _samples_from_list(doc.get("samples", []), "samples")
    implicit = _samples_from_list(doc.get("implicit_samples", []), "implicit_samples")
    dims = {s.features.shape[0] for s in [*samples, *implicit]}
    if model.is_fitted:
        dims.add(model.n_features)
    if len(dims) > 1:
        raise ValueError(f"inconsistent feature lengths {sorted(dims)}")

    report = doc.get("report", {})
    return CalibrationData(
        backend=_str(doc["backend"], "backend"),
        feature_version=_str(doc["feature_version"], "feature_version"),
        layout_signature=_str(doc["layout_signature"], "layout_signature"),
        monitors=[_monitor_from_dict(m) for m in _list(doc["monitors"], "monitors")],
        samples=samples,
        implicit_samples=implicit,
        model=model,
        report=report if isinstance(report, dict) else {},
        created_at=str(doc.get("created_at", "")),
    )


# ------------------------------------------------------------------- monitors
def _monitor_to_dict(m: Monitor) -> dict[str, Any]:
    return {
        "index": m.index,
        "name": m.name,
        "rect": m.rect.to_list(),
        "primary": m.primary,
        "scale": m.scale,
    }


def _monitor_from_dict(d: Any) -> Monitor:
    if not isinstance(d, dict):
        raise ValueError("monitor entry is not an object")
    return Monitor(
        index=int(d["index"]),
        name=str(d.get("name", "")),
        rect=Rect.from_list(d["rect"]),
        primary=bool(d.get("primary", False)),
        scale=float(d.get("scale", 1.0)),
    )


# -------------------------------------------------------------------- samples
def _samples_to_list(samples: Iterable[CalibrationSample]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    dropped = 0
    for s in samples:
        features = np.asarray(s.features, dtype=np.float64).reshape(-1)
        values = [*features.tolist(), s.x, s.y, s.weight]
        if not all(math.isfinite(v) for v in values):
            dropped += 1
            continue
        out.append(
            {
                "features": [round(v, _SAMPLE_DECIMALS) for v in features.tolist()],
                "x": round(float(s.x), _SAMPLE_DECIMALS),
                "y": round(float(s.y), _SAMPLE_DECIMALS),
                "monitor": int(s.monitor_index),
                "point": int(s.point_id),
                "weight": round(float(s.weight), _SAMPLE_DECIMALS),
            }
        )
    if dropped:
        log.warning("Not saving %d calibration samples with non-finite values", dropped)
    return out


def _samples_from_list(items: Any, name: str) -> list[CalibrationSample]:
    out = []
    for item in _list(items, name):
        if not isinstance(item, dict):
            raise ValueError(f"{name}: entry is not an object")
        features = np.asarray(item["features"], dtype=np.float64)
        if features.ndim != 1 or features.size == 0:
            raise ValueError(f"{name}: features must be a non-empty list of numbers")
        sample = CalibrationSample(
            features=features,
            x=float(item["x"]),
            y=float(item["y"]),
            monitor_index=int(item["monitor"]),
            point_id=int(item["point"]),
            weight=float(item.get("weight", 1.0)),
        )
        if not (
            np.all(np.isfinite(features))
            and math.isfinite(sample.x)
            and math.isfinite(sample.y)
            and math.isfinite(sample.weight)
            and sample.weight >= 0
        ):
            raise ValueError(f"{name}: invalid numbers")
        out.append(sample)
    return out


# ------------------------------------------------------------------- helpers
def _round_model(d: dict[str, Any]) -> dict[str, Any]:
    def rnd(v: Any) -> Any:
        if isinstance(v, float):
            return float(f"{v:.{_MODEL_DIGITS}g}")
        if isinstance(v, list):
            return [rnd(x) for x in v]
        return v

    return {k: rnd(v) for k, v in d.items()}


def _json_safe(value: Any) -> Any:
    """Plain JSON types only: numpy scalars unwrapped, NaN/Infinity as null."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dumps(doc: dict[str, Any]) -> str:
    """Pretty JSON with the bulky parts (samples, model arrays) one item per line."""

    def compact(value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), allow_nan=False, ensure_ascii=False)

    lines = []
    for key, value in doc.items():
        if key in ("samples", "implicit_samples", "monitors") and value:
            body = "[\n" + ",\n".join(f"    {compact(v)}" for v in value) + "\n  ]"
        elif key == "model" and value:
            items = ",\n".join(f"    {json.dumps(k)}: {compact(v)}" for k, v in value.items())
            body = "{\n" + items + "\n  }"
        else:
            body = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)
            body = body.replace("\n", "\n  ")
        lines.append(f"  {json.dumps(key)}: {body}")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value
