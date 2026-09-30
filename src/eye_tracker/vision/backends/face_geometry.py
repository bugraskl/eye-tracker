"""MediaPipe's canonical face model, read from the Face Landmarker bundle.

``geometry_pipeline_metadata_landmarks.binarypb`` is a member of MediaPipe's
``face_landmarker.task`` bundle (Apache-2.0, see ``models/NOTICE.md``). It is a
serialised ``GeometryPipelineMetadata`` protocol buffer, defined in
``mediapipe/tasks/cc/vision/face_geometry/proto/geometry_pipeline_metadata.proto``
and ``mesh_3d.proto``::

    message GeometryPipelineMetadata {
      optional InputSource input_source = 3;
      optional Mesh3d canonical_mesh = 1;
      repeated WeightedLandmarkRef procrustes_landmark_basis = 2;
    }
    message WeightedLandmarkRef { optional uint32 landmark_id = 1; optional float weight = 2; }
    message Mesh3d {
      optional VertexType vertex_type = 1;        // VERTEX_PT = 0: x, y, z, u, v per vertex
      optional PrimitiveType primitive_type = 2;  // TRIANGLE = 0
      repeated float vertex_buffer = 3;
      repeated uint32 index_buffer = 4;
    }

Only the canonical vertex positions and the Procrustes landmark basis (the
landmarks MediaPipe itself uses to fit the rigid head pose, with weights) are
needed, so a few lines of protobuf wire-format decoding replace a protobuf
dependency.

The canonical mesh uses centimetres in a head-centred frame: ``x`` points to the
right of a camera image showing the face (the subject's left), ``y`` up and
``z`` out of the face towards the camera.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Landmarks in the canonical mesh (the landmark model adds 10 iris points).
MESH_VERTICES = 468
_FLOATS_PER_VERTEX = 5  # VERTEX_PT: position (3) + texture coordinates (2)

# Protobuf wire types.
_VARINT, _I64, _LEN, _I32 = 0, 1, 2, 5


class GeometryError(ValueError):
    """The geometry file is missing, truncated or not the expected message."""


@dataclass(frozen=True, slots=True)
class FaceGeometry:
    """Canonical face vertices and the landmark basis for rigid pose fitting."""

    #: ``(468, 3)`` canonical vertex positions in centimetres (see module docs).
    vertices: np.ndarray
    #: Landmark indices MediaPipe fits the head pose to (stable, rigid points).
    basis_ids: tuple[int, ...]
    #: Relative weight of each basis landmark in MediaPipe's Procrustes fit.
    basis_weights: np.ndarray


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf) or shift > 63:
            raise GeometryError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """Yield ``(field number, wire type, value)`` for every field of a message."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        value: int | bytes
        if wire == _VARINT:
            value, pos = _varint(buf, pos)
        elif wire == _I64:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == _LEN:
            size, pos = _varint(buf, pos)
            value, pos = buf[pos : pos + size], pos + size
        elif wire == _I32:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise GeometryError(f"unsupported protobuf wire type {wire}")
        if pos > len(buf):
            raise GeometryError("truncated field")
        yield number, wire, value


def _floats(wire: int, value: int | bytes) -> list[float]:
    """Decode a ``repeated float`` entry, packed or not."""
    if not isinstance(value, bytes) or wire not in (_I32, _LEN) or len(value) % 4:
        raise GeometryError("malformed float field")
    return list(struct.unpack(f"<{len(value) // 4}f", value))


def parse_face_geometry(data: bytes) -> FaceGeometry:
    """Decode a serialised ``GeometryPipelineMetadata`` message.

    Raises:
        GeometryError: The data is not a valid landmark geometry message.
    """
    vertex_buffer: list[float] = []
    basis: list[tuple[int, float]] = []
    for number, wire, value in _fields(data):
        if number == 1 and wire == _LEN and isinstance(value, bytes):  # canonical_mesh
            for sub, sub_wire, sub_value in _fields(value):
                if sub == 1 and sub_value != 0:
                    raise GeometryError(f"unsupported mesh vertex type {sub_value!r}")
                if sub == 3:
                    vertex_buffer += _floats(sub_wire, sub_value)
        elif number == 2 and wire == _LEN and isinstance(value, bytes):  # landmark basis
            landmark_id, weight = 0, 0.0
            for sub, sub_wire, sub_value in _fields(value):
                if sub == 1 and isinstance(sub_value, int):
                    landmark_id = sub_value
                elif sub == 2:
                    values = _floats(sub_wire, sub_value)
                    if len(values) != 1:
                        raise GeometryError("malformed landmark weight")
                    weight = values[0]
            basis.append((landmark_id, weight))

    if len(vertex_buffer) != MESH_VERTICES * _FLOATS_PER_VERTEX:
        raise GeometryError(
            f"expected {MESH_VERTICES} mesh vertices, found "
            f"{len(vertex_buffer) / _FLOATS_PER_VERTEX:g}"
        )
    vertices = np.asarray(vertex_buffer, dtype=np.float64).reshape(-1, _FLOATS_PER_VERTEX)[:, :3]
    if not np.all(np.isfinite(vertices)):
        raise GeometryError("non-finite mesh vertices")
    basis = [(i, w) for i, w in basis if 0 <= i < MESH_VERTICES and w > 0.0]
    if len(basis) < 6:
        raise GeometryError(f"only {len(basis)} usable landmarks in the Procrustes basis")
    return FaceGeometry(
        vertices=vertices,
        basis_ids=tuple(i for i, _ in basis),
        basis_weights=np.array([w for _, w in basis], dtype=np.float64),
    )


def load_face_geometry(path: Path) -> FaceGeometry:
    """Read and decode the geometry file at ``path``.

    Raises:
        GeometryError: The file cannot be read or decoded.
    """
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise GeometryError(f"cannot read {path}: {exc}") from exc
    return parse_face_geometry(data)
