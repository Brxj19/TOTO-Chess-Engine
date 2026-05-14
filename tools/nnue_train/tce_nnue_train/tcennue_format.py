"""Binary format helpers for TCE-owned `.tcennue` files."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAGIC = b"TCENNUE\0"
FORMAT_VERSION = 1
CHECKSUM_SIZE = 32
HEADER_STRUCT = struct.Struct("<8sIIIIIIdIQ")


@dataclass(frozen=True)
class TcennueHeader:
    format_version: int
    feature_count: int
    half_dim: int
    hidden1_dim: int
    hidden2_dim: int
    output_dim: int
    target_scale: float
    tensor_count: int
    metadata_size: int

    def pack(self) -> bytes:
        return HEADER_STRUCT.pack(
            MAGIC,
            self.format_version,
            self.feature_count,
            self.half_dim,
            self.hidden1_dim,
            self.hidden2_dim,
            self.output_dim,
            self.target_scale,
            self.tensor_count,
            self.metadata_size,
        )


def unpack_header(data: bytes) -> TcennueHeader:
    if len(data) < HEADER_STRUCT.size:
        raise ValueError("file is too small to contain a tcennue header")

    (
        magic,
        format_version,
        feature_count,
        half_dim,
        hidden1_dim,
        hidden2_dim,
        output_dim,
        target_scale,
        tensor_count,
        metadata_size,
    ) = HEADER_STRUCT.unpack(data[: HEADER_STRUCT.size])

    if magic != MAGIC:
        raise ValueError("invalid tcennue magic bytes")

    return TcennueHeader(
        format_version=format_version,
        feature_count=feature_count,
        half_dim=half_dim,
        hidden1_dim=hidden1_dim,
        hidden2_dim=hidden2_dim,
        output_dim=output_dim,
        target_scale=target_scale,
        tensor_count=tensor_count,
        metadata_size=metadata_size,
    )


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_tcennue(path: str | Path) -> tuple[TcennueHeader, dict[str, Any], bytes, bytes]:
    file_bytes = Path(path).read_bytes()
    header = unpack_header(file_bytes)
    metadata_start = HEADER_STRUCT.size
    metadata_end = metadata_start + header.metadata_size
    checksum_start = len(file_bytes) - CHECKSUM_SIZE
    if metadata_end > checksum_start:
        raise ValueError("metadata extends beyond payload/checksum boundary")

    metadata = json.loads(file_bytes[metadata_start:metadata_end].decode("utf-8"))
    payload = file_bytes[metadata_end:checksum_start]
    checksum = file_bytes[checksum_start:]
    return header, metadata, payload, checksum
