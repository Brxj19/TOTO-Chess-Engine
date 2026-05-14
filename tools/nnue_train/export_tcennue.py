#!/usr/bin/env python3
"""Export or inspect TCE-owned `.tcennue` NNUE binaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tce_nnue_train.export import export_tcennue, inspect_tcennue  # noqa: E402
from tce_nnue_train.quantize import quantize_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained TCE NNUE checkpoint to .tcennue or inspect one."
    )
    parser.add_argument("--checkpoint", type=Path, help="Input trained .pt checkpoint.")
    parser.add_argument("--output", type=Path, help="Output .tcennue path.")
    parser.add_argument("--inspect", type=Path, help="Inspect an existing .tcennue file.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.inspect:
        if args.checkpoint or args.output:
            raise ValueError("--inspect cannot be combined with --checkpoint or --output")
        return
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --inspect is used")
    if args.output is None:
        raise ValueError("--output is required unless --inspect is used")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.inspect:
        print(json.dumps(inspect_tcennue(args.inspect), indent=2, sort_keys=True))
        return

    qnet = quantize_checkpoint(args.checkpoint)
    result = export_tcennue(qnet, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
