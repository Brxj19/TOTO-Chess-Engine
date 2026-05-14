#!/usr/bin/env python3
"""Dump sparse inference vectors for C `.tcennue` parity tests."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tce_nnue_train.dataset import TceNnueFeatureDataset  # noqa: E402
from tce_nnue_train.quantize import QuantizedNetwork, quantize_checkpoint  # noqa: E402
from tce_nnue_train.tcennue_format import read_tcennue  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump sparse test vectors for C-side NNUE inference parity."
    )
    parser.add_argument("--data", required=True, type=Path, help="Sparse feature .npz.")
    parser.add_argument("--checkpoint", type=Path, help="Trained .pt checkpoint.")
    parser.add_argument("--tcennue", type=Path, help="Exported .tcennue file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file.")
    parser.add_argument("--samples", type=int, default=16, help="Number of vectors.")
    parser.add_argument("--seed", type=int, default=1, help="Sample seed.")
    parser.add_argument(
        "--first",
        action="store_true",
        help="Dump the first N rows instead of deterministic random rows.",
    )
    return parser.parse_args()


def tensor_dtype(dtype: str) -> np.dtype:
    if dtype == "int16":
        return np.dtype("<i2")
    if dtype == "int32":
        return np.dtype("<i4")
    raise ValueError(f"unsupported tensor dtype: {dtype}")


def load_quantized_from_tcennue(path: Path) -> QuantizedNetwork:
    header, metadata, payload, _checksum = read_tcennue(path)
    tensors: dict[str, np.ndarray] = {}
    for tensor in metadata["tensors"]:
        start = int(tensor["offset"])
        end = start + int(tensor["size"])
        dtype = tensor_dtype(tensor["dtype"])
        array = np.frombuffer(payload[start:end], dtype=dtype)
        tensors[tensor["name"]] = array.reshape(tensor["shape"]).copy()

    metadata.setdefault("feature_count", header.feature_count)
    metadata.setdefault("half_dim", header.half_dim)
    metadata.setdefault("hidden1_dim", header.hidden1_dim)
    metadata.setdefault("hidden2_dim", header.hidden2_dim)
    metadata.setdefault("output_dim", header.output_dim)
    metadata.setdefault("target_scale", header.target_scale)
    return QuantizedNetwork(tensors=tensors, metadata=metadata)


def load_quantized(args: argparse.Namespace) -> QuantizedNetwork:
    if args.tcennue:
        return load_quantized_from_tcennue(args.tcennue)
    if args.checkpoint:
        return quantize_checkpoint(args.checkpoint)
    raise ValueError("provide --checkpoint or --tcennue")


def clipped_relu(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def predict_cp(
    qnet: QuantizedNetwork,
    white_features: list[int],
    black_features: list[int],
    side_to_move: int,
) -> int:
    tensors = qnet.tensors
    scales = qnet.metadata["weight_scales"]
    target_scale = float(qnet.metadata["target_scale"])

    ft = tensors["ft_weight"].astype(np.int32)
    white_acc = ft[np.asarray(white_features, dtype=np.int64)].sum(axis=0)
    black_acc = ft[np.asarray(black_features, dtype=np.int64)].sum(axis=0)
    if side_to_move == 0:
        first, second = white_acc, black_acc
    else:
        first, second = black_acc, white_acc
    x_int = np.concatenate([first, second]).astype(np.int64)

    h1_w = tensors["hidden1_weight"].astype(np.int64)
    h1_b = tensors["hidden1_bias"].astype(np.int64)
    h1 = clipped_relu((h1_w @ x_int + h1_b).astype(np.float64) * scales["hidden1_bias"])

    h2_w = tensors["hidden2_weight"].astype(np.float64) * scales["hidden2_weight"]
    h2_b = tensors["hidden2_bias"].astype(np.float64) * scales["hidden2_bias"]
    h2 = clipped_relu(h2_w @ h1 + h2_b)

    out_w = tensors["output_weight"].astype(np.float64) * scales["output_weight"]
    out_b = tensors["output_bias"].astype(np.float64) * scales["output_bias"]
    normalized = float((out_w @ h2 + out_b)[0])
    return int(round(normalized * target_scale))


def sample_indices(size: int, samples: int, seed: int, first: bool) -> list[int]:
    if samples < 1:
        raise ValueError("--samples must be at least 1")
    count = min(samples, size)
    if first:
        return list(range(count))
    rng = random.Random(seed)
    return sorted(rng.sample(range(size), count))


def main() -> None:
    args = parse_args()
    qnet = load_quantized(args)
    dataset = TceNnueFeatureDataset(
        args.data,
        target_scale=float(qnet.metadata.get("target_scale", 1000.0)),
    )
    vectors: list[dict[str, Any]] = []

    for index in sample_indices(len(dataset), args.samples, args.seed, args.first):
        sample = dataset[index]
        white_features = sample["white_features"].tolist()
        black_features = sample["black_features"].tolist()
        side_to_move = int(sample["side_to_move"])
        vectors.append(
            {
                "index": index,
                "fen": sample["fen"],
                "side_to_move": side_to_move,
                "white_features": white_features,
                "black_features": black_features,
                "target_cp": int(round(float(sample["eval_cp"]))),
                "expected_pred_cp": predict_cp(
                    qnet,
                    white_features,
                    black_features,
                    side_to_move,
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(vectors, output_file, indent=2)
        output_file.write("\n")
    print(f"Wrote {len(vectors)} inference vectors to {args.output}")


if __name__ == "__main__":
    main()
