"""Dataset loader for TCE NNUE sparse feature .npz files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = {
    "white_features",
    "white_offsets",
    "black_features",
    "black_offsets",
    "eval_cp",
    "side_to_move",
}


@dataclass(frozen=True)
class NnueBatch:
    white_features: torch.Tensor
    white_offsets: torch.Tensor
    black_features: torch.Tensor
    black_offsets: torch.Tensor
    side_to_move: torch.Tensor
    target: torch.Tensor
    eval_cp: torch.Tensor
    best_move: list[str]
    depth: torch.Tensor
    fen: list[str]


class TceNnueFeatureDataset(Dataset):
    """Map-style dataset backed by arrays produced by build_features.py."""

    def __init__(self, path: str | Path, target_scale: float = 1000.0) -> None:
        if target_scale <= 0:
            raise ValueError("target_scale must be positive")

        self.path = Path(path)
        self.target_scale = float(target_scale)
        data = np.load(self.path, allow_pickle=False)
        try:
            missing = REQUIRED_KEYS.difference(data.files)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"{self.path} is missing required keys: {names}")

            self.white_features = data["white_features"].astype(np.int64, copy=True)
            self.white_offsets = data["white_offsets"].astype(np.int64, copy=True)
            self.black_features = data["black_features"].astype(np.int64, copy=True)
            self.black_offsets = data["black_offsets"].astype(np.int64, copy=True)
            self.eval_cp = data["eval_cp"].astype(np.float32, copy=True)
            self.side_to_move = data["side_to_move"].astype(np.int64, copy=True)
            self.feature_count = int(data["feature_count"][0]) if "feature_count" in data else None
            self.best_move = data["best_move"].astype(str, copy=True) if "best_move" in data else None
            self.depth = data["depth"].astype(np.int16, copy=True) if "depth" in data else None
            self.fens = data["fens"].astype(str, copy=True) if "fens" in data else None
        finally:
            data.close()

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        row_count = len(self.eval_cp)
        if len(self.side_to_move) != row_count:
            raise ValueError("side_to_move length does not match eval_cp length")
        if len(self.white_offsets) != row_count + 1:
            raise ValueError("white_offsets length must equal row_count + 1")
        if len(self.black_offsets) != row_count + 1:
            raise ValueError("black_offsets length must equal row_count + 1")
        if self.best_move is not None and len(self.best_move) != row_count:
            raise ValueError("best_move length does not match eval_cp length")
        if self.depth is not None and len(self.depth) != row_count:
            raise ValueError("depth length does not match eval_cp length")
        if self.fens is not None and len(self.fens) != row_count:
            raise ValueError("fens length does not match eval_cp length")

    def __len__(self) -> int:
        return len(self.eval_cp)

    def __getitem__(self, index: int) -> dict[str, Any]:
        white_start = self.white_offsets[index]
        white_end = self.white_offsets[index + 1]
        black_start = self.black_offsets[index]
        black_end = self.black_offsets[index + 1]

        target = self.eval_cp[index] / self.target_scale
        target = float(np.clip(target, -2.0, 2.0))

        return {
            "white_features": torch.as_tensor(
                self.white_features[white_start:white_end], dtype=torch.long
            ),
            "black_features": torch.as_tensor(
                self.black_features[black_start:black_end], dtype=torch.long
            ),
            "side_to_move": int(self.side_to_move[index]),
            "target": target,
            "eval_cp": float(self.eval_cp[index]),
            "best_move": str(self.best_move[index]) if self.best_move is not None else "",
            "depth": int(self.depth[index]) if self.depth is not None else -1,
            "fen": str(self.fens[index]) if self.fens is not None else "",
        }


def _pack_feature_lists(feature_lists: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    chunks = []
    total = 0
    for features in feature_lists:
        features = features.to(dtype=torch.long)
        chunks.append(features)
        total += int(features.numel())
        offsets.append(total)

    if chunks:
        flat = torch.cat(chunks) if total else torch.empty(0, dtype=torch.long)
    else:
        flat = torch.empty(0, dtype=torch.long)

    return flat, torch.tensor(offsets, dtype=torch.long)


def collate_feature_batch(samples: list[dict[str, Any]]) -> NnueBatch:
    white_features, white_offsets = _pack_feature_lists(
        [sample["white_features"] for sample in samples]
    )
    black_features, black_offsets = _pack_feature_lists(
        [sample["black_features"] for sample in samples]
    )

    return NnueBatch(
        white_features=white_features,
        white_offsets=white_offsets,
        black_features=black_features,
        black_offsets=black_offsets,
        side_to_move=torch.tensor(
            [sample["side_to_move"] for sample in samples], dtype=torch.long
        ),
        target=torch.tensor([sample["target"] for sample in samples], dtype=torch.float32),
        eval_cp=torch.tensor([sample["eval_cp"] for sample in samples], dtype=torch.float32),
        best_move=[sample["best_move"] for sample in samples],
        depth=torch.tensor([sample["depth"] for sample in samples], dtype=torch.int16),
        fen=[sample["fen"] for sample in samples],
    )
