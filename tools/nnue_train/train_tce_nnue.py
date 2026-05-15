#!/usr/bin/env python3
"""Train TCE NNUE with validation logging and early stopping."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tce_nnue_train.dataset import TceNnueFeatureDataset, collate_feature_batch  # noqa: E402
from tce_nnue_train.features import FEATURE_COUNT  # noqa: E402
from tce_nnue_train.model import TceNnueModel  # noqa: E402
from tce_nnue_train.train import forward_model, move_batch_to_device, resolve_device  # noqa: E402


@dataclass(frozen=True)
class TrainArgs:
    data: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    val_ratio: float
    target_scale: float
    half_dim: int
    device: str
    seed: int
    num_workers: int
    patience: int
    min_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TCE-owned NNUE model.")
    parser.add_argument("--data", required=True, help="Input sparse feature .npz file.")
    parser.add_argument("--output-dir", required=True, help="Training run directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--target-scale", type=float, default=1000.0)
    parser.add_argument("--half-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: TrainArgs) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.target_scale <= 0:
        raise ValueError("--target-scale must be positive")
    if args.half_dim < 1:
        raise ValueError("--half-dim must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative")


def split_dataset(dataset: TceNnueFeatureDataset, val_ratio: float, seed: int):
    val_size = int(len(dataset) * val_ratio)
    if val_ratio > 0 and val_size == 0 and len(dataset) > 1:
        val_size = 1
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("training split is empty")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_feature_batch,
        generator=generator,
    )


def run_epoch(model, loader, optimizer, loss_fn, device, target_scale: float) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_abs = 0.0
    total = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = forward_model(model, batch)
        loss = loss_fn(pred, batch.target)
        loss.backward()
        optimizer.step()

        n = int(batch.target.numel())
        total += n
        total_loss += float(loss.item()) * n
        total_abs += float(torch.sum(torch.abs(pred.detach() - batch.target)).item())
    return total_loss / max(1, total), (total_abs / max(1, total)) * target_scale


@torch.no_grad()
def validate(model, loader, loss_fn, device, target_scale: float) -> tuple[float, float]:
    if len(loader.dataset) == 0:
        return 0.0, 0.0
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    total = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred = forward_model(model, batch)
        loss = loss_fn(pred, batch.target)
        n = int(batch.target.numel())
        total += n
        total_loss += float(loss.item()) * n
        total_abs += float(torch.sum(torch.abs(pred - batch.target)).item())
    return total_loss / max(1, total), (total_abs / max(1, total)) * target_scale


def save_checkpoint(path: Path, epoch: int, model, optimizer, args: TrainArgs, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(args),
            "model_config": model.config(),
            "metrics": metrics,
        },
        path,
    )


def append_log(path: Path, row: dict[str, float | int]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "train_loss", "val_loss", "train_mae_cp", "val_mae_cp"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ns = parse_args()
    args = TrainArgs(**vars(ns))
    validate_args(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    if log_path.exists():
        log_path.unlink()

    dataset = TceNnueFeatureDataset(args.data, target_scale=args.target_scale)
    train_set, val_set = split_dataset(dataset, args.val_ratio, args.seed)
    train_loader = make_loader(train_set, args.batch_size, True, args.seed, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.seed, args.num_workers)

    model = TceNnueModel(
        feature_count=dataset.feature_count or FEATURE_COUNT,
        half_dim=args.half_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    with (output_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump({"training": asdict(args), "model": model.config()}, config_file, indent=2)
        config_file.write("\n")

    best_val = float("inf")
    stale_epochs = 0
    print(f"Training on {device} with {len(train_set)} train / {len(val_set)} validation positions")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, optimizer, loss_fn, device, args.target_scale)
        val_loss, val_mae = validate(model, val_loader, loss_fn, device, args.target_scale)
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae_cp": train_mae,
            "val_mae_cp": val_mae,
        }
        append_log(output_dir / "training_log.csv", {"epoch": epoch, **metrics})
        save_checkpoint(output_dir / "checkpoints" / "latest.pt", epoch, model, optimizer, args, metrics)

        improved = val_loss < best_val - args.min_delta
        if improved:
            best_val = val_loss
            stale_epochs = 0
            save_checkpoint(output_dir / "checkpoints" / "best.pt", epoch, model, optimizer, args, metrics)
        else:
            stale_epochs += 1

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_mae_cp={train_mae:.2f} val_mae_cp={val_mae:.2f}"
        )

        if args.patience and stale_epochs >= args.patience:
            print(f"early stopping after {epoch} epochs")
            break


if __name__ == "__main__":
    main()
