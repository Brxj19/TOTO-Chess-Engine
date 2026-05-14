#!/usr/bin/env python3
"""Quantize a trained TCE NNUE checkpoint and print tensor metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tce_nnue_train.quantize import quantize_checkpoint, quantized_forward_check  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a trained TCE NNUE checkpoint for export."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Input .pt checkpoint.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a simple dequantized Python inference sanity check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qnet = quantize_checkpoint(args.checkpoint)
    summary = {
        "checkpoint": str(args.checkpoint),
        "metadata": qnet.metadata,
        "tensors": {
            name: {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": int(tensor.nbytes),
            }
            for name, tensor in qnet.tensors.items()
        },
    }
    if args.check:
        summary["dequantized_check_output"] = quantized_forward_check(qnet, [0])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
