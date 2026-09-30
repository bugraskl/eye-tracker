"""Tests for the canonical face geometry reader (a minimal protobuf decoder)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from eye_tracker import paths
from eye_tracker.vision.backends.face_geometry import (
    MESH_VERTICES,
    GeometryError,
    load_face_geometry,
    parse_face_geometry,
)

GEOMETRY_FILE = "geometry_pipeline_metadata_landmarks.binarypb"


# ------------------------------------------------------------ protobuf encoder
def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _key(number: int, wire: int) -> bytes:
    return _varint(number << 3 | wire)


def _len_field(number: int, payload: bytes) -> bytes:
    return _key(number, 2) + _varint(len(payload)) + payload


def _mesh(vertices: np.ndarray, *, packed: bool = True, vertex_type: int = 0) -> bytes:
    """A ``Mesh3d`` message with VERTEX_PT vertices (u, v set to zero)."""
    buffer = np.zeros((len(vertices), 5), np.float32)
    buffer[:, :3] = vertices
    floats = buffer.ravel().tolist()
    body = _key(1, 0) + _varint(vertex_type)
    if packed:
        body += _len_field(3, struct.pack(f"<{len(floats)}f", *floats))
    else:
        body += b"".join(_key(3, 5) + struct.pack("<f", f) for f in floats)
    return body + _key(4, 0) + _varint(7)  # an index-buffer entry, ignored


def _landmark(landmark_id: int, weight: float) -> bytes:
    return _key(1, 0) + _varint(landmark_id) + _key(2, 5) + struct.pack("<f", weight)


def _message(vertices: np.ndarray, basis: list[tuple[int, float]], **mesh_kwargs: object) -> bytes:
    out = _key(3, 0) + _varint(1)  # input_source, ignored
    out += _len_field(1, _mesh(vertices, **mesh_kwargs))  # type: ignore[arg-type]
    for landmark_id, weight in basis:
        out += _len_field(2, _landmark(landmark_id, weight))
    return out


BASIS = [(4, 0.07), (6, 0.12), (33, 0.05), (133, 0.05), (263, 0.05), (362, 0.05), (1, 0.01)]


def _vertices() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.normal(0.0, 5.0, (MESH_VERTICES, 3)).astype(np.float32)


# -------------------------------------------------------------------- parsing
@pytest.mark.parametrize("packed", [True, False])
def test_parse_round_trip(packed: bool) -> None:
    vertices = _vertices()
    geometry = parse_face_geometry(_message(vertices, BASIS, packed=packed))
    np.testing.assert_allclose(geometry.vertices, vertices, rtol=1e-6)
    assert geometry.basis_ids == tuple(i for i, _ in BASIS)
    np.testing.assert_allclose(geometry.basis_weights, [w for _, w in BASIS], rtol=1e-6)


def test_unusable_basis_entries_are_dropped() -> None:
    basis = [*BASIS, (MESH_VERTICES + 3, 0.5), (10, 0.0)]
    geometry = parse_face_geometry(_message(_vertices(), basis))
    assert geometry.basis_ids == tuple(i for i, _ in BASIS)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "expected 468 mesh vertices"),
        (b"\x80", "truncated varint"),
        (_key(9, 3), "unsupported protobuf wire type"),
        (_key(1, 2) + _varint(50) + b"abc", "truncated field"),
    ],
)
def test_malformed_data(data: bytes, message: str) -> None:
    with pytest.raises(GeometryError, match=message):
        parse_face_geometry(data)


def test_wrong_vertex_type() -> None:
    with pytest.raises(GeometryError, match="vertex type"):
        parse_face_geometry(_message(_vertices(), BASIS, vertex_type=1))


def test_too_few_vertices() -> None:
    with pytest.raises(GeometryError, match="expected 468"):
        parse_face_geometry(_message(_vertices()[:100], BASIS))


def test_too_small_basis() -> None:
    with pytest.raises(GeometryError, match="Procrustes basis"):
        parse_face_geometry(_message(_vertices(), BASIS[:3]))


def test_non_finite_vertices() -> None:
    vertices = _vertices()
    vertices[5, 1] = np.nan
    with pytest.raises(GeometryError, match="non-finite"):
        parse_face_geometry(_message(vertices, BASIS))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GeometryError, match="cannot read"):
        load_face_geometry(tmp_path / "missing.binarypb")


# ---------------------------------------------------------- the bundled model
def test_bundled_canonical_face() -> None:
    geometry = load_face_geometry(paths.model_path(GEOMETRY_FILE))
    v = geometry.vertices
    assert v.shape == (MESH_VERTICES, 3)
    assert np.all(np.isfinite(v))
    # Centimetres, head-centred: a face is ~15 cm wide and symmetric about x = 0.
    assert 10.0 < v[:, 0].max() - v[:, 0].min() < 20.0
    assert abs(v[:, 0].mean()) < 0.5
    # Orientation: the subject's right eye corner (33) is on the image left,
    # y points up (forehead above chin) and z towards the camera (nose tip in front).
    assert v[33, 0] < 0.0 < v[263, 0]
    assert v[10, 1] > v[152, 1]
    assert v[4, 2] > v[:, 2].mean()
    # MediaPipe fits the pose to a set of stable landmarks, with positive weights.
    ids = geometry.basis_ids
    assert 6 <= len(ids) < MESH_VERTICES
    assert len(set(ids)) == len(ids)
    assert np.all(geometry.basis_weights > 0.0)
