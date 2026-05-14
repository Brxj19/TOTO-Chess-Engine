"""Training loop for the baseline TCE NNUE model."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from .dataset import NnueBatch, TceNnueFeatureDataset, collate_feature_batch
from .features import FEATURE_COUNT
from .model import TceNnueModel


@dataclass(frozen=True)
class TrainConfig:
    data: str
    output_dir: str
    epochs: int = 1
    batch_size: int = 256
    lr: float = 0.001
    val_split: float = 0.1
    target_scale: float = 1000.0
    half_dim: int = 128
    device: str = "auto"
    seed: int = 1
    num_workers: int = 0


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: NnueBatch, device: torch.device) -> NnueBatch:
    return NnueBatch(
        white_features=batch.white_features.to(device),
        white_offsets=batch.white_offsets.to(device),
        black_features=batch.black_features.to(device),
        black_offsets=batch.black_offsets.to(device),
        side_to_move=batch.side_to_move.to(device),
        target=batch.target.to(device),
        eval_cp=batch.eval_cp.to(device),
        best_move=batch.best_move,
        depth=batch.depth.to(device),
        fen=batch.fen,
    )


def make_splits(
    dataset: TceNnueFeatureDataset,
    val_split: float,
    seed: int,
) -> tuple[Subset, Subset]:
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0.0, 1.0)")
    if len(dataset) < 1:
        raise ValueError("dataset is empty")

    val_size = int(len(dataset) * val_split)
    if val_split > 0 and val_size == 0 and len(dataset) > 1:
        val_size = 1
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("training split is empty")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    return train_dataset, val_dataset


def make_loader(
    dataset: Subset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_feature_batch,
        generator=generator,
    )


def forward_model(model: TceNnueModel, batch: NnueBatch) -> torch.Tensor:
    return model(
        batch.white_features,
        batch.white_offsets,
        batch.black_features,
        batch.black_offsets,
        batch.side_to_move,
    )


def train_one_epoch(
    model: TceNnueModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_model(model, batch)
        loss = loss_fn(prediction, batch.target)
        loss.backward()
        optimizer.step()

        batch_size = int(batch.target.numel())
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: TceNnueModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    target_scale: float,
) -> tuple[float, float]:
    if len(loader.dataset) == 0:
        return 0.0, 0.0

    model.eval()
    total_loss = 0.0
    total_abs_cp = 0.0
    total_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        prediction = forward_model(model, batch)
        loss = loss_fn(prediction, batch.target)

        batch_size = int(batch.target.numel())
        total_loss += float(loss.item()) * batch_size
        total_abs_cp += float(torch.sum(torch.abs(prediction - batch.target)).item())
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    mae_cp = (total_abs_cp / max(1, total_samples)) * target_scale
    return avg_loss, mae_cp


def save_checkpoint(
    path: Path,
    model: TceNnueModel,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "model_config": model.config(),
            "metrics": metrics,
        },
        path,
    )


def write_config(output_dir: Path, config: TrainConfig, model: TceNnueModel) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "training": asdict(config),
        "model": model.config(),
        "feature_count": FEATURE_COUNT,
        "target_key": "eval_cp",
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=2)
        config_file.write("\n")


def append_metrics(
    metrics_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_mae_cp: float,
) -> None:
    exists = metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "train_loss", "val_loss", "val_mae_cp"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae_cp": val_mae_cp,
            }
        )


def train(config: TrainConfig) -> None:
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.lr <= 0:
        raise ValueError("lr must be positive")
    if config.target_scale <= 0:
        raise ValueError("target_scale must be positive")
    if config.half_dim < 1:
        raise ValueError("half_dim must be at least 1")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.csv"

    dataset = TceNnueFeatureDataset(config.data, target_scale=config.target_scale)
    train_dataset, val_dataset = make_splits(dataset, config.val_split, config.seed)
    train_loader = make_loader(
        train_dataset,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = make_loader(
        val_dataset,
        config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    model = TceNnueModel(
        feature_count=dataset.feature_count or FEATURE_COUNT,
        half_dim=config.half_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = nn.SmoothL1Loss()

    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        metrics_path.unlink()
    write_config(output_dir, config, model)

    best_val_loss = float("inf")
    print(
        f"Training on {device} with {len(train_dataset)} train and "
        f"{len(val_dataset)} validation positions"
    )

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mae_cp = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            config.target_scale,
        )
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae_cp": val_mae_cp,
        }

        print(
            f"epoch {epoch}/{config.epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_mae_cp={val_mae_cp:.2f}"
        )
        append_metrics(metrics_path, epoch, train_loss, val_loss, val_mae_cp)

        save_checkpoint(
            checkpoints_dir / "last.pt",
            model,
            optimizer,
            config,
            epoch,
            metrics,
        )
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoints_dir / "best.pt",
                model,
                optimizer,
                config,
                epoch,
                metrics,
            )
