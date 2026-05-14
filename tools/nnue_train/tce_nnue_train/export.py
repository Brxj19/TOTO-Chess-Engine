"""Export and inspect TCE-owned `.tcennue` binaries."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .quantize import TENSOR_ORDER, QuantizedNetwork
from .tcennue_format import (
    CHECKSUM_SIZE,
    FORMAT_VERSION,
    HEADER_STRUCT,
    TcennueHeader,
    canonical_json_bytes,
    read_tcennue,
)


DTYPE_NAMES = {
    np.dtype("<i2"): "int16",
    np.dtype("<i4"): "int32",
}


def tensor_bytes(tensor: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(tensor)
    return contiguous.tobytes(order="C")


def tensor_dtype_name(tensor: np.ndarray) -> str:
    dtype = np.dtype(tensor.dtype).newbyteorder("<")
    return DTYPE_NAMES.get(dtype, str(dtype))


def build_payload_and_metadata(qnet: QuantizedNetwork) -> tuple[bytes, dict[str, Any]]:
    payload_parts: list[bytes] = []
    tensor_entries = []
    offset = 0

    for name in TENSOR_ORDER:
        if name not in qnet.tensors:
            raise ValueError(f"quantized network is missing tensor {name}")

        tensor = np.ascontiguousarray(qnet.tensors[name])
        raw = tensor_bytes(tensor)
        tensor_hash = hashlib.sha256(raw).hexdigest()
        tensor_entries.append(
            {
                "name": name,
                "dtype": tensor_dtype_name(tensor),
                "shape": list(tensor.shape),
                "offset": offset,
                "size": len(raw),
                "sha256": tensor_hash,
            }
        )
        payload_parts.append(raw)
        offset += len(raw)

    payload = b"".join(payload_parts)
    metadata = dict(qnet.metadata)
    metadata.update(
        {
            "format_version": FORMAT_VERSION,
            "header_size": HEADER_STRUCT.size,
            "checksum": "sha256(header+metadata+payload)",
            "payload_size": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "tensors": tensor_entries,
        }
    )
    return payload, metadata


def export_tcennue(qnet: QuantizedNetwork, output_path: str | Path) -> dict[str, Any]:
    payload, metadata = build_payload_and_metadata(qnet)
    metadata_bytes = canonical_json_bytes(metadata)
    header = TcennueHeader(
        format_version=FORMAT_VERSION,
        feature_count=int(metadata["feature_count"]),
        half_dim=int(metadata["half_dim"]),
        hidden1_dim=int(metadata["hidden1_dim"]),
        hidden2_dim=int(metadata["hidden2_dim"]),
        output_dim=int(metadata["output_dim"]),
        target_scale=float(metadata["target_scale"]),
        tensor_count=len(TENSOR_ORDER),
        metadata_size=len(metadata_bytes),
    )

    body = header.pack() + metadata_bytes + payload
    checksum = hashlib.sha256(body).digest()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(body + checksum)

    return {
        "path": str(output),
        "size": output.stat().st_size,
        "checksum_sha256": checksum.hex(),
        "payload_sha256": metadata["payload_sha256"],
        "tensor_count": len(TENSOR_ORDER),
    }


def inspect_tcennue(path: str | Path) -> dict[str, Any]:
    file_bytes = Path(path).read_bytes()
    header, metadata, payload, checksum = read_tcennue(path)
    expected = hashlib.sha256(file_bytes[:-CHECKSUM_SIZE]).digest()
    checksum_ok = checksum == expected
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    payload_ok = payload_sha256 == metadata.get("payload_sha256")

    tensors = metadata.get("tensors", [])
    for tensor in tensors:
        start = int(tensor["offset"])
        end = start + int(tensor["size"])
        if start < 0 or end > len(payload):
            raise ValueError(f"tensor {tensor.get('name')} is outside the payload")
        actual_hash = hashlib.sha256(payload[start:end]).hexdigest()
        if actual_hash != tensor.get("sha256"):
            raise ValueError(f"tensor {tensor.get('name')} checksum mismatch")

    return {
        "path": str(path),
        "file_size": len(file_bytes),
        "checksum_ok": checksum_ok,
        "payload_ok": payload_ok,
        "header": {
            "format_version": header.format_version,
            "feature_count": header.feature_count,
            "half_dim": header.half_dim,
            "hidden1_dim": header.hidden1_dim,
            "hidden2_dim": header.hidden2_dim,
            "output_dim": header.output_dim,
            "target_scale": header.target_scale,
            "tensor_count": header.tensor_count,
            "metadata_size": header.metadata_size,
        },
        "metadata": metadata,
        "checksum_sha256": checksum.hex(),
        "payload_sha256": payload_sha256,
    }
