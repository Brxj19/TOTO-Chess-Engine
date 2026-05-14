#!/usr/bin/env python3
"""Inspect a trained TCE NNUE checkpoint on sample dataset positions."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tce_nnue_train.dataset import TceNnueFeatureDataset, collate_feature_batch  # noqa: E402
from tce_nnue_train.features import FEATURE_COUNT  # noqa: E402
from tce_nnue_train.model import TceNnueModel  # noqa: E402
from tce_nnue_train.train import forward_model, move_batch_to_device, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained TCE NNUE checkpoint and evaluate sample positions."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to .pt file.")
    parser.add_argument("--data", required=True, type=Path, help="Sparse feature .npz file.")
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Number of positions to sample from the dataset.",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=None,
        help="Override checkpoint target scale for converting predictions to CP.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device: auto, cpu, cuda, cuda:0, mps, etc.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random sample seed.")
    parser.add_argument(
        "--first",
        action="store_true",
        help="Use the first N positions instead of random sampling.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} did not contain a checkpoint dictionary")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is missing model_state_dict")
    return checkpoint


def checkpoint_target_scale(checkpoint: dict[str, Any], override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--target-scale must be positive")
        return override

    config = checkpoint.get("config", {})
    target_scale = float(config.get("target_scale", 1000.0))
    if target_scale <= 0:
        raise ValueError("checkpoint target_scale must be positive")
    return target_scale


def checkpoint_model_config(
    checkpoint: dict[str, Any],
    dataset: TceNnueFeatureDataset,
) -> dict[str, int]:
    model_config = dict(checkpoint.get("model_config", {}))
    model_config.setdefault("feature_count", dataset.feature_count or FEATURE_COUNT)
    model_config.setdefault("half_dim", 128)
    model_config.setdefault("hidden1_dim", 64)
    model_config.setdefault("hidden2_dim", 32)
    return model_config


def sample_indices(size: int, samples: int, seed: int, first: bool) -> list[int]:
    if samples < 1:
        raise ValueError("--samples must be at least 1")
    count = min(samples, size)
    if first:
        return list(range(count))

    rng = random.Random(seed)
    return sorted(rng.sample(range(size), count))


def format_fen(fen: str, max_len: int = 96) -> str:
    if len(fen) <= max_len:
        return fen
    return fen[: max_len - 3] + "..."


@torch.no_grad()
def validate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    target_scale = checkpoint_target_scale(checkpoint, args.target_scale)

    dataset = TceNnueFeatureDataset(args.data, target_scale=target_scale)
    indices = sample_indices(len(dataset), args.samples, args.seed, args.first)
    model_config = checkpoint_model_config(checkpoint, dataset)

    model = TceNnueModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = [dataset[index] for index in indices]
    batch = collate_feature_batch(samples)
    batch = move_batch_to_device(batch, device)
    predictions = forward_model(model, batch).cpu() * target_scale
    targets = batch.eval_cp.cpu()
    errors = predictions - targets
    mae = float(torch.mean(torch.abs(errors)).item()) if len(indices) else 0.0

    print(f"checkpoint: {args.checkpoint}")
    print(f"data:       {args.data}")
    print(f"device:     {device}")
    print(f"samples:    {len(indices)}")
    print(f"mae_cp:     {mae:.2f}")
    print()
    print("index\ttarget_cp\tpred_cp\terror_cp\tdepth\tbest_move\tfen")

    for row, index in enumerate(indices):
        print(
            f"{index}\t"
            f"{float(targets[row]):.1f}\t"
            f"{float(predictions[row]):.1f}\t"
            f"{float(errors[row]):.1f}\t"
            f"{int(batch.depth[row])}\t"
            f"{batch.best_move[row]}\t"
            f"{format_fen(batch.fen[row])}"
        )


def main() -> None:
    args = parse_args()
    validate(args)


if __name__ == "__main__":
    main()
