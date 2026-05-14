#!/usr/bin/env python3
"""Train the baseline TCE NNUE model from sparse feature datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tce_nnue_train.train import TrainConfig, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline NNUE-like model for TOTO Chess Engine."
    )
    parser.add_argument("--data", required=True, help="Input sparse feature .npz file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for checkpoints, metrics.csv, and config.json.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation fraction, deterministically split.",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1000.0,
        help="Divide eval_cp by this value before training.",
    )
    parser.add_argument(
        "--half-dim",
        type=int,
        default=128,
        help="Feature transformer half dimension.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, mps, etc.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        target_scale=args.target_scale,
        half_dim=args.half_dim,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    train(config)


if __name__ == "__main__":
    main()
