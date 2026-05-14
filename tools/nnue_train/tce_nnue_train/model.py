"""Baseline PyTorch NNUE-like model for TCE sparse features."""

from __future__ import annotations

import torch
from torch import nn

from .features import FEATURE_COUNT


class ClippedReLU(nn.Module):
    def __init__(self, maximum: float = 1.0) -> None:
        super().__init__()
        self.maximum = maximum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, self.maximum)


class TceNnueModel(nn.Module):
    """Small baseline NNUE-style network.

    A shared sparse feature transformer embeds white and black perspective
    feature IDs. The forward pass concatenates side-to-move perspective first
    and opponent perspective second before dense layers.
    """

    def __init__(
        self,
        feature_count: int = FEATURE_COUNT,
        half_dim: int = 128,
        hidden1_dim: int = 64,
        hidden2_dim: int = 32,
        activation: str = "clipped_relu",
    ) -> None:
        super().__init__()
        self.feature_count = int(feature_count)
        self.half_dim = int(half_dim)
        self.hidden1_dim = int(hidden1_dim)
        self.hidden2_dim = int(hidden2_dim)

        self.feature_transformer = nn.EmbeddingBag(
            self.feature_count,
            self.half_dim,
            mode="sum",
            include_last_offset=True,
        )

        if activation == "relu":
            act: nn.Module = nn.ReLU()
        elif activation == "clipped_relu":
            act = ClippedReLU()
        else:
            raise ValueError(f"unsupported activation: {activation}")

        self.network = nn.Sequential(
            nn.Linear(self.half_dim * 2, self.hidden1_dim),
            act,
            nn.Linear(self.hidden1_dim, self.hidden2_dim),
            act,
            nn.Linear(self.hidden2_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_transformer.weight, mean=0.0, std=0.01)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(
        self,
        white_features: torch.Tensor,
        white_offsets: torch.Tensor,
        black_features: torch.Tensor,
        black_offsets: torch.Tensor,
        side_to_move: torch.Tensor,
    ) -> torch.Tensor:
        white_acc = self.feature_transformer(white_features, white_offsets)
        black_acc = self.feature_transformer(black_features, black_offsets)

        white_to_move = side_to_move == 0
        stm_acc = torch.where(white_to_move.unsqueeze(1), white_acc, black_acc)
        opponent_acc = torch.where(white_to_move.unsqueeze(1), black_acc, white_acc)

        x = torch.cat([stm_acc, opponent_acc], dim=1)
        return self.network(x).squeeze(1)

    def config(self) -> dict[str, int]:
        return {
            "feature_count": self.feature_count,
            "half_dim": self.half_dim,
            "hidden1_dim": self.hidden1_dim,
            "hidden2_dim": self.hidden2_dim,
        }
