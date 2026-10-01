"""Persistence of calibrations as a small, human-readable JSON file.

Only numbers are stored: the feature vectors (head pose angles, head position,
iris offsets), the screen points they were labelled with and the fitted model.
Camera frames are never part of a calibration.

A calibration is only valid for the setup it was recorded on: the monitor
layout, the vision backend (and the version of its measurements) and the
camera. A laptop that moves between desks therefore needs one calibration per
desk, so the file holds a small :class:`CalibrationLibrary` of *profiles*, one
per setup, most recently used first. Version 1 files, which held a single
calibration, are read as a library with one profile.

The file is pretty-printed with one sample per line so it stays readable and
reasonably small (a two-monitor calibration is roughly 50 KB). Loading never
raises: a missing, unreadable, corrupt or too new file yields an empty library
(``None`` from :func:`load_calibration`) and the app asks for a new calibration;
a single corrupt profile is skipped and the others are kept.
"""

from __future__ import annotations

import json
import logging
import math
import weakref
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import atomic_write_text
from ..types import Monitor, Rect, layout_signature, virtual_bounds
from .calibration import (
    CalibrationSample,
    axis_errors_from_dict,
    per_axis_errors,
    samples_to_arrays,
)
from .model import SUPPORTED_DEGREES, GazeModel, lopo_predictions

log = logging.getLogger(__name__)

#: Version 2: a list of profiles. Version 1 (a single calibration) is still read.
CALIBRATION_VERSION = 2

#: Marker that identifies the file type.
FILE_FORMAT = "eye-tracker-calibration"

#: Profiles kept by a :class:`CalibrationLibrary`; the least recently used go first.
MAX_PROFILES = 8

#: Camera frames whose aspect ratios differ by more than this (relative) show a
#: different field of view. The resolution alone does not matter: every backend
#: normalises its measurements by the frame size.
ASPECT_TOLERANCE = 0.01

# Samples are measurements, so six decimals are far below their noise. Model
# parameters keep ten significant digits so a reloaded model predicts the same
# pixels as the one that was saved.
_SAMPLE_DECIMALS = 6
_MODEL_DIGITS = 10

# Everything a malformed (hand-edited or damaged) file can raise while being
# parsed: json.JSONDecodeError is a ValueError, int(inf) an OverflowError and
# very deep nesting a RecursionError.
_PARSE_ERRORS = (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError)

#: ``CalibrationData.key``: (layout signature, backend, feature version, camera).
ProfileKey = tuple[str, str, str, str]

#: Backend names of earlier versions whose measurements did not change with the
#: new name: the OpenCV/YuNet backend is now called ``lite`` (same features, same
#: ``feature_version``), so its calibrations stay valid. The MediaPipe backend's
#: successor, ``facemesh``, measures differently (another ``feature_version``),
#: so ``mediapipe`` calibrations are deliberately not carried over.
BACKEND_ALIASES: dict[str, str] = {"opencv": "lite"}


def canonical_backend(name: str) -> str:
    """The current name of a vision backend (see :data:`BACKEND_ALIASES`)."""
    return BACKEND_ALIASES.get(name, name)


def utc_now_iso() -> str:
    """Current time as ISO 8601 UTC with second precision, e.g. ``2026-09-30T12:00:00+00:00``."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(eq=False)
class CalibrationData:
    """A calibration and everything needed to decide whether it is still valid.

    ``camera`` is the camera device setting (``CameraSettings.device``, e.g.
    ``"0"``) and ``frame_size`` the ``(w, h)`` of the camera frames the samples
    were measured on (``CalibrationCollector.frame_size``). Head pose and face
    position are measured relative to the camera, so another camera, or the same
    one with another field of view, needs another calibration. ``""`` and
    ``(0, 0)`` mean unknown (files written before they were recorded) and are
    never a reason to reject the calibration.
    """

    backend: str
    feature_version: str
    layout_signature: str
    monitors: list[Monitor]
    samples: list[CalibrationSample]
    implicit_samples: list[CalibrationSample]
    model: GazeModel
    report: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    camera: str = ""
    frame_size: tuple[int, int] = (0, 0)

    @property
    def key(self) -> ProfileKey:
        """The setup this calibration belongs to; a library keeps one profile per key.

        The backend is given by its current name (:func:`canonical_backend`), so a
        profile of the renamed ``opencv`` backend and a new ``lite`` one are the same.
        """
        return (
            self.layout_signature,
            canonical_backend(self.backend),
            self.feature_version,
            self.camera.strip(),
        )

    def is_compatible(
        self,
        backend_name: str,
        feature_version: str,
        monitors: Sequence[Monitor],
        camera: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> tuple[bool, str]:
        """``(True, "")`` if usable with this backend, monitor layout and camera, else
        ``(False, reason)`` where ``reason`` is a short lower-case phrase such as
        ``"the monitor layout changed"``.

        ``camera`` and ``frame_size`` are the current camera device setting and
        frame size; ``None``, ``""`` or ``(0, 0)`` (here or stored) skip that check.
        Backend names are compared by their current names (:func:`canonical_backend`).
        """
        current, stored = canonical_backend(backend_name), canonical_backend(self.backend)
        if current != stored:
            return (
                False,
                f"calibrated with the {stored} backend, but {current} is in use",
            )
        if feature_version != self.feature_version:
            return False, (
                f"the {backend_name} backend's measurements changed "
                f"({self.feature_version} → {feature_version})"
            )
        if layout_signature(monitors) != self.layout_signature:
            return False, "the monitor layout changed"
        ok, reason = self.camera_matches(camera, frame_size)
        if not ok:
            return False, reason
        if not self.model.is_fitted:
            return False, "the calibration has no fitted model"
        return True, ""

    def camera_matches(
        self, camera: str | None = None, frame_size: tuple[int, int] | None = None
    ) -> tuple[bool, str]:
        """The camera part of :meth:`is_compatible`, e.g. for the first frame after
        the camera was (re)opened: ``(False, "the camera changed")`` for another
        device, ``(False, "the camera's aspect ratio changed (…)")`` for another
        field of view. Unknown values on either side pass."""
        stored, current = self.camera.strip(), (camera or "").strip()
        if stored and current and stored != current:
            return False, "the camera changed"
        if frame_size is not None and not same_aspect(self.frame_size, frame_size):
            (w0, h0), (w1, h1) = self.frame_size, frame_size
            return False, f"the camera's aspect ratio changed ({w0}x{h0} → {w1}x{h1})"
        return True, ""

    @property
    def grade(self) -> str | None:
        """Quality grade from the report (``"excellent"`` … ``"poor"``), if known."""
        value = self.report.get("grade")
        return value if isinstance(value, str) else None


def same_aspect(a: Sequence[int], b: Sequence[int]) -> bool:
    """Whether frame sizes ``a`` and ``b`` ``(w, h)`` have the same aspect ratio
    (within :data:`ASPECT_TOLERANCE`); True if either is unknown (not positive)."""
    (wa, ha), (wb, hb) = a, b
    if min(wa, ha, wb, hb) <= 0:
        return True
    return abs((wa / ha) / (wb / hb) - 1.0) <= ASPECT_TOLERANCE


# ------------------------------------------------------------ gaze accuracy
#: Per-axis errors already worked out for a calibration (loaded, recomputed or
#: found impossible to recompute: ``{}``), so each profile is looked at once.
_AXIS_ERRORS: weakref.WeakKeyDictionary[CalibrationData, dict[int, tuple[float, float]]] = (
    weakref.WeakKeyDictionary()
)


def axis_error(data: CalibrationData, monitor: Monitor) -> tuple[float, float] | None:
    """How far off the gaze estimate typically is on ``monitor``: ``(x_px, y_px)``.

    The answer comes, in this order, from

    1. the report's ``per_monitor_error_px`` (calibrations made by this version),
    2. the leave-one-point-out predictions recomputed from the calibration's own
       samples with the report's ``alpha`` and ``degree`` (older calibrations;
       the result is added to :attr:`CalibrationData.report`, so it is saved
       with the profile's next write), and
    3. the report's mean error divided by √2 on both axes.

    ``monitor`` is matched to the calibration's monitors by its rectangle (Qt may
    number the same screens differently after a restart), then by index.
    ``None`` when nothing is known.

    The first call for an old profile recomputes on the calling thread (the GUI
    thread, from the controller), on purpose: the leave-one-point-out
    predictions cost one design matrix and one small ``p x p`` solve per
    calibration dot, about 2 ms for a two-monitor calibration of 360 samples and
    4 ms for 960 (degree 3, measured on a desktop; ``tests/test_store.py`` keeps
    a generous bound). It happens once per profile and session; later calls are
    dictionary lookups. A background thread would cost more than it saves.
    """
    errors = _AXIS_ERRORS.get(data)
    if errors is None:
        errors = _axis_errors_of(data)
        _AXIS_ERRORS[data] = errors
    index = next((m.index for m in data.monitors if m.rect == monitor.rect), monitor.index)
    found = errors.get(index)
    if found is not None:
        return found
    mean = data.report.get("mean_error_px")
    if isinstance(mean, int | float) and not isinstance(mean, bool):
        value = float(mean) / math.sqrt(2.0)
        if math.isfinite(value) and value > 0.0:
            return (value, value)
    return None


def _axis_errors_of(data: CalibrationData) -> dict[int, tuple[float, float]]:
    try:
        stored = axis_errors_from_dict(data.report.get("per_monitor_error_px"))
    except ValueError:
        stored = {}
    if stored:
        return stored
    computed = _recompute_axis_errors(data)
    if computed:
        data.report["per_monitor_error_px"] = {str(k): list(v) for k, v in computed.items()}
        log.debug("Gaze error per axis recomputed for the calibration of %s", data.created_at)
    return computed


def _recompute_axis_errors(data: CalibrationData) -> dict[int, tuple[float, float]]:
    """Leave-one-point-out errors per monitor and axis of an older calibration."""
    known = {m.index for m in data.monitors}
    samples = [s for s in data.samples if s.monitor_index in known]
    groups = np.array([s.point_id for s in samples])
    if not samples or np.unique(groups).shape[0] < 2:
        return {}
    report = data.report
    alpha = report.get("alpha")
    degree = report.get("degree")
    if not isinstance(alpha, int | float) or isinstance(alpha, bool) or not alpha >= 0:
        alpha = data.model.alpha
    if not isinstance(degree, int) or isinstance(degree, bool) or degree not in SUPPORTED_DEGREES:
        degree = data.model.degree
    try:
        X, Y, W = samples_to_arrays(samples)
        preds = lopo_predictions(
            X,
            Y,
            groups,
            float(alpha),
            degree=int(degree),
            bounds=virtual_bounds(data.monitors),
            weights=W,
            nonlinear=data.model.nonlinear,
        )
        return per_axis_errors(preds, Y, [s.monitor_index for s in samples])
    except (ValueError, np.linalg.LinAlgError) as exc:
        log.debug("Cannot recompute the per-axis gaze error: %s", exc)
        return {}


# --------------------------------------------------------------------- library
class CalibrationLibrary:
    """The saved calibrations, one profile per setup (:attr:`CalibrationData.key`),
    most recently used first.

    Typical use by the engine::

        library = CalibrationLibrary.load(path)
        data = library.best(backend, feature_version, monitors, camera=device)
        ...after a calibration or a refit:   library.put(data); library.save(path)
        ...after switching to a profile:     library.mark_used(data); library.save(path)

    At most ``max_profiles`` profiles are kept; adding one more drops the least
    recently used. Profiles are held by reference, so changing a profile's model
    or implicit samples in place and calling :meth:`save` persists the change.
    """

    def __init__(
        self, profiles: Iterable[CalibrationData] = (), *, max_profiles: int = MAX_PROFILES
    ) -> None:
        if max_profiles < 1:
            raise ValueError("max_profiles must be >= 1")
        self.max_profiles = int(max_profiles)
        self._profiles: list[CalibrationData] = []
        # Given most recent first; adding in reverse keeps that order and drops
        # later duplicates of a key.
        for data in reversed(list(profiles)):
            self.put(data)

    # ------------------------------------------------------------- persistence
    @classmethod
    def load(cls, path: Path, *, max_profiles: int = MAX_PROFILES) -> CalibrationLibrary:
        """Read the calibration file; never raises (a bad file gives an empty library)."""
        return cls(_read_profiles(Path(path)), max_profiles=max_profiles)

    def save(self, path: Path) -> None:
        """Write every profile to ``path`` atomically. Raises ``OSError`` if writing fails."""
        doc = {
            "format": FILE_FORMAT,
            "version": CALIBRATION_VERSION,
            "profiles": [_profile_to_doc(p) for p in self._profiles],
        }
        atomic_write_text(Path(path), _dumps(doc))
        log.debug("Saved %d calibration profile(s) to %s", len(self._profiles), path)

    # ------------------------------------------------------------- access
    @property
    def profiles(self) -> list[CalibrationData]:
        """All profiles, most recently used first."""
        return list(self._profiles)

    @property
    def latest(self) -> CalibrationData | None:
        """The most recently used profile (``None`` when empty)."""
        return self._profiles[0] if self._profiles else None

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self) -> Iterator[CalibrationData]:
        return iter(list(self._profiles))

    def __contains__(self, data: object) -> bool:
        return any(p is data for p in self._profiles)

    def get(self, key: ProfileKey) -> CalibrationData | None:
        """The profile stored for ``key`` (see :attr:`CalibrationData.key`)."""
        return next((p for p in self._profiles if p.key == key), None)

    def match(
        self,
        backend_name: str,
        feature_version: str,
        monitors: Sequence[Monitor],
        camera: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> tuple[CalibrationData | None, str]:
        """The best profile for the current setup, or ``(None, reason)``.

        Among compatible profiles (:meth:`CalibrationData.is_compatible`), one
        whose camera and frame size are known to match beats one where they are
        unknown; otherwise the most recently used wins. Without any, ``reason``
        explains what is wrong with the most recently used profile
        (``"not calibrated yet"`` for an empty library).
        """
        best: tuple[int, int, CalibrationData] | None = None
        for position, data in enumerate(self._profiles):
            ok, _ = data.is_compatible(backend_name, feature_version, monitors, camera, frame_size)
            if not ok:
                continue
            specific = int(bool(data.camera.strip() and (camera or "").strip()))
            specific += int(frame_size is not None and min(*data.frame_size, *frame_size) > 0)
            if best is None or (-specific, position) < (-best[0], best[1]):
                best = (specific, position, data)
        if best is not None:
            return best[2], ""
        latest = self.latest
        if latest is None:
            return None, "not calibrated yet"
        _, reason = latest.is_compatible(
            backend_name, feature_version, monitors, camera, frame_size
        )
        return None, reason

    def best(
        self,
        backend_name: str,
        feature_version: str,
        monitors: Sequence[Monitor],
        camera: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> CalibrationData | None:
        """The profile :meth:`match` chooses, or ``None``."""
        return self.match(backend_name, feature_version, monitors, camera, frame_size)[0]

    # ------------------------------------------------------------- changes
    def put(self, data: CalibrationData) -> list[CalibrationData]:
        """Add ``data`` as the most recently used profile; returns the profiles it evicted.

        It replaces the profile with the same :attr:`~CalibrationData.key`, and a
        profile of the same setup whose camera is unknown (``""``, from an older
        file), which was most likely made with the same camera. Profiles beyond
        ``max_profiles`` are then dropped, least recently used first.
        """
        layout, backend, version, camera = data.key
        replaced = [
            p
            for p in self._profiles
            if p is data
            or p.key == data.key
            or (camera and p.key[:3] == (layout, backend, version) and not p.key[3])
        ]
        self._profiles = [data, *(p for p in self._profiles if p not in replaced)]
        evicted = self._profiles[self.max_profiles :]
        del self._profiles[self.max_profiles :]
        for p in evicted:
            log.info("Dropping the least recently used calibration (%s)", p.created_at)
        return [p for p in replaced if p is not data] + evicted

    def mark_used(self, data: CalibrationData) -> bool:
        """Make a stored profile the most recently used; False if it is not stored."""
        if data not in self:
            return False
        self._profiles = [data, *(p for p in self._profiles if p is not data)]
        return True

    def remove(self, data: CalibrationData) -> bool:
        """Remove a stored profile; False if it is not stored."""
        if data not in self:
            return False
        self._profiles = [p for p in self._profiles if p is not data]
        return True

    def clear(self) -> None:
        """Remove every profile."""
        self._profiles.clear()


# ------------------------------------------------------------ single profile
def save_calibration(path: Path, data: CalibrationData) -> None:
    """Store ``data`` in the calibration file at ``path`` as the most recently used
    profile, replacing the profile of the same setup and keeping the others.

    Writes atomically; raises ``OSError`` if writing fails. A caller that holds a
    :class:`CalibrationLibrary` should use :meth:`CalibrationLibrary.put` and
    :meth:`~CalibrationLibrary.save` instead, which avoids re-reading the file.
    """
    library = CalibrationLibrary.load(path)
    library.put(data)
    library.save(path)


def load_calibration(path: Path) -> CalibrationData | None:
    """The most recently used calibration in the file at ``path``; ``None`` if the file
    is missing, corrupt or unsupported. Never raises."""
    return CalibrationLibrary.load(path).latest


# ------------------------------------------------------------------ reading
class _NewerVersion(Exception):
    pass


def _read_profiles(path: Path) -> list[CalibrationData]:
    try:
        # utf-8-sig: a file saved by an editor (or PowerShell 5.1) with a BOM is
        # still the user's calibration, not a corrupt file.
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("Could not read calibration %s: %s", path, exc)
        return []
    try:
        doc = json.loads(text)
        return _profiles_from_doc(doc, path)
    except _NewerVersion as exc:
        log.warning("Calibration %s was written by a newer Eye Tracker (%s); ignored", path, exc)
    except _PARSE_ERRORS as exc:
        log.warning("Ignoring corrupt calibration file %s: %s", path, exc)
    return []


def _profiles_from_doc(doc: Any, path: Path) -> list[CalibrationData]:
    if not isinstance(doc, dict):
        raise ValueError("root is not an object")
    if doc.get("format", FILE_FORMAT) != FILE_FORMAT:
        raise ValueError(f"not a calibration file (format {doc.get('format')!r})")
    version = doc.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"invalid version {version!r}")
    if version > CALIBRATION_VERSION:
        raise _NewerVersion(f"version {version}")
    if version == 1:
        return [_profile_from_doc(doc)]

    profiles = []
    for i, item in enumerate(_list(doc["profiles"], "profiles")):
        try:
            profiles.append(_profile_from_doc(item))
        except _PARSE_ERRORS as exc:
            log.warning("Ignoring corrupt calibration profile %d in %s: %s", i + 1, path, exc)
    return profiles


def _profile_from_doc(doc: Any) -> CalibrationData:
    if not isinstance(doc, dict):
        raise ValueError("profile is not an object")
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
        # Stored under the current name, so the next save writes that.
        backend=canonical_backend(_str(doc["backend"], "backend")),
        feature_version=_str(doc["feature_version"], "feature_version"),
        layout_signature=_str(doc["layout_signature"], "layout_signature"),
        monitors=[_monitor_from_dict(m) for m in _list(doc["monitors"], "monitors")],
        samples=samples,
        implicit_samples=implicit,
        model=model,
        report=report if isinstance(report, dict) else {},
        created_at=str(doc.get("created_at", "")),
        camera=_camera_from_doc(doc.get("camera")),
        frame_size=_frame_size_from_doc(doc.get("frame_size")),
    )


def _camera_from_doc(value: Any) -> str:
    # Optional metadata: a malformed value means "unknown", not a lost calibration.
    return value if isinstance(value, str) else ""


def _frame_size_from_doc(value: Any) -> tuple[int, int]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value)
    ):
        return (value[0], value[1])
    return (0, 0)


# ------------------------------------------------------------------ writing
def _profile_to_doc(data: CalibrationData) -> dict[str, Any]:
    width, height = data.frame_size
    return {
        "created_at": data.created_at,
        "backend": data.backend,
        "feature_version": data.feature_version,
        "layout_signature": data.layout_signature,
        "camera": data.camera,
        "frame_size": [int(width), int(height)],
        "monitors": [_monitor_to_dict(m) for m in data.monitors],
        "report": _json_safe(data.report),
        "model": _round_model(data.model.to_dict()),
        "samples": _samples_to_list(data.samples),
        "implicit_samples": _samples_to_list(data.implicit_samples),
    }


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
    scale = float(d.get("scale", 1.0))
    if not math.isfinite(scale):
        raise ValueError("monitor scale is not a finite number")
    return Monitor(
        index=int(d["index"]),
        name=str(d.get("name", "")),
        rect=Rect.from_list(d["rect"]),
        primary=bool(d.get("primary", False)),
        scale=scale,
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
        if isinstance(v, dict):  # the look-away regions
            return {k: rnd(x) for k, x in v.items()}
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


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def _dumps(doc: dict[str, Any]) -> str:
    """Pretty JSON: each profile's bulky parts (samples, model arrays) one item per line."""
    lines = []
    for key, value in doc.items():
        if key == "profiles":
            items = ",\n".join("    " + _dump_profile(p, "    ") for p in value)
            body = "[\n" + items + "\n  ]" if value else "[]"
        else:
            body = json.dumps(value, allow_nan=False, ensure_ascii=False)
        lines.append(f"  {json.dumps(key)}: {body}")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def _dump_profile(profile: dict[str, Any], indent: str) -> str:
    inner = indent + "  "
    lines = []
    for key, value in profile.items():
        if key in ("samples", "implicit_samples", "monitors") and value:
            rows = ",\n".join(f"{inner}  {_compact(v)}" for v in value)
            body = "[\n" + rows + f"\n{inner}]"
        elif key == "model" and value:
            rows = ",\n".join(f"{inner}  {json.dumps(k)}: {_compact(v)}" for k, v in value.items())
            body = "{\n" + rows + f"\n{inner}}}"
        elif key == "frame_size":
            body = _compact(value)
        else:
            body = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)
            body = body.replace("\n", "\n" + inner)
        lines.append(f"{inner}{json.dumps(key)}: {body}")
    return "{\n" + ",\n".join(lines) + f"\n{indent}}}"


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value
