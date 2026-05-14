"""Quantize trained TCE NNUE checkpoints into integer tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import FEATURE_COUNT
from .model import TceNnueModel


INT16_MAX = 32767
INT32_MAX = 2147483647
INT32_QUANT_MAX = 2147483000
DEFAULT_ACTIVATION_MIN = 0.0
DEFAULT_ACTIVATION_MAX = 1.0


TENSOR_ORDER = [
    "ft_weight",
    "hidden1_weight",
    "hidden1_bias",
    "hidden2_weight",
    "hidden2_bias",
    "output_weight",
    "output_bias",
]


@dataclass(frozen=True)
class QuantizedNetwork:
    tensors: dict[str, np.ndarray]
    metadata: dict[str, Any]


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} did not contain a checkpoint dictionary")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is missing model_state_dict")
    return checkpoint


def checkpoint_target_scale(checkpoint: dict[str, Any]) -> float:
    config = checkpoint.get("config", {})
    target_scale = float(config.get("target_scale", 1000.0))
    if target_scale <= 0:
        raise ValueError("checkpoint target_scale must be positive")
    return target_scale


def checkpoint_model_config(checkpoint: dict[str, Any]) -> dict[str, int]:
    model_config = dict(checkpoint.get("model_config", {}))
    model_config.setdefault("feature_count", FEATURE_COUNT)
    model_config.setdefault("half_dim", 128)
    model_config.setdefault("hidden1_dim", 64)
    model_config.setdefault("hidden2_dim", 32)
    return {
        "feature_count": int(model_config["feature_count"]),
        "half_dim": int(model_config["half_dim"]),
        "hidden1_dim": int(model_config["hidden1_dim"]),
        "hidden2_dim": int(model_config["hidden2_dim"]),
    }


def load_model_from_checkpoint(checkpoint: dict[str, Any]) -> TceNnueModel:
    model = TceNnueModel(**checkpoint_model_config(checkpoint))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def symmetric_int16_scale(values: np.ndarray) -> float:
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs == 0.0:
        return 1.0
    return max_abs / INT16_MAX


def quantize_int16(values: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.nan_to_num(values / scale, nan=0.0, posinf=INT16_MAX, neginf=-INT16_MAX)
    quantized = np.rint(scaled)
    quantized = np.clip(quantized, -INT16_MAX, INT16_MAX)
    return quantized.astype("<i2", copy=False)


def quantize_bias_int32(values: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("bias scale must be positive")
    scaled = np.asarray(values, dtype=np.float64) / float(scale)
    scaled = np.nan_to_num(
        scaled,
        nan=0.0,
        posinf=INT32_QUANT_MAX,
        neginf=-INT32_QUANT_MAX,
    )
    quantized = np.rint(scaled)
    quantized = np.clip(quantized, -INT32_QUANT_MAX, INT32_QUANT_MAX)
    return quantized.astype("<i4", copy=False)


def cpu_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=True)


def dense_layers(model: TceNnueModel) -> tuple[torch.nn.Linear, torch.nn.Linear, torch.nn.Linear]:
    hidden1 = model.network[0]
    hidden2 = model.network[2]
    output = model.network[4]
    if not all(isinstance(layer, torch.nn.Linear) for layer in (hidden1, hidden2, output)):
        raise ValueError("unexpected TceNnueModel dense layer layout")
    return hidden1, hidden2, output


def quantize_model(model: TceNnueModel, target_scale: float) -> QuantizedNetwork:
    hidden1, hidden2, output = dense_layers(model)

    ft_weight_f = cpu_numpy(model.feature_transformer.weight)
    hidden1_weight_f = cpu_numpy(hidden1.weight)
    hidden2_weight_f = cpu_numpy(hidden2.weight)
    output_weight_f = cpu_numpy(output.weight)

    scales = {
        "ft_weight": symmetric_int16_scale(ft_weight_f),
        "hidden1_weight": symmetric_int16_scale(hidden1_weight_f),
        "hidden2_weight": symmetric_int16_scale(hidden2_weight_f),
        "output_weight": symmetric_int16_scale(output_weight_f),
    }

    # Bias scales describe the quantized dot-product domain for each layer.
    scales["hidden1_bias"] = scales["ft_weight"] * scales["hidden1_weight"]
    scales["hidden2_bias"] = scales["hidden1_weight"] * scales["hidden2_weight"]
    scales["output_bias"] = scales["hidden2_weight"] * scales["output_weight"]

    tensors = {
        "ft_weight": quantize_int16(ft_weight_f, scales["ft_weight"]),
        "hidden1_weight": quantize_int16(hidden1_weight_f, scales["hidden1_weight"]),
        "hidden1_bias": quantize_bias_int32(cpu_numpy(hidden1.bias), scales["hidden1_bias"]),
        "hidden2_weight": quantize_int16(hidden2_weight_f, scales["hidden2_weight"]),
        "hidden2_bias": quantize_bias_int32(cpu_numpy(hidden2.bias), scales["hidden2_bias"]),
        "output_weight": quantize_int16(output_weight_f, scales["output_weight"]),
        "output_bias": quantize_bias_int32(cpu_numpy(output.bias), scales["output_bias"]),
    }

    metadata = {
        "quantization_version": 1,
        "target_scale": target_scale,
        "feature_count": model.feature_count,
        "half_dim": model.half_dim,
        "hidden1_dim": model.hidden1_dim,
        "hidden2_dim": model.hidden2_dim,
        "output_dim": 1,
        "activation": "clipped_relu",
        "activation_min": DEFAULT_ACTIVATION_MIN,
        "activation_max": DEFAULT_ACTIVATION_MAX,
        "weight_scales": scales,
        "tensor_order": TENSOR_ORDER,
        "dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
    }
    return QuantizedNetwork(tensors=tensors, metadata=metadata)


def quantize_checkpoint(path: str | Path) -> QuantizedNetwork:
    checkpoint = load_checkpoint(path)
    model = load_model_from_checkpoint(checkpoint)
    return quantize_model(model, checkpoint_target_scale(checkpoint))


def dequantize(tensor: np.ndarray, scale: float) -> np.ndarray:
    return tensor.astype(np.float32) * np.float32(scale)


def quantized_forward_check(qnet: QuantizedNetwork, active_features: list[int]) -> float:
    """Simple float-domain check using dequantized exported tensors.

    This is not the eventual engine inference path; it catches obvious tensor
    extraction, shape, and scale mistakes before export.
    """

    md = qnet.metadata
    scales = md["weight_scales"]
    ft = dequantize(qnet.tensors["ft_weight"], scales["ft_weight"])
    h1_w = dequantize(qnet.tensors["hidden1_weight"], scales["hidden1_weight"])
    h1_b = qnet.tensors["hidden1_bias"].astype(np.float32) * np.float32(scales["hidden1_bias"])
    h2_w = dequantize(qnet.tensors["hidden2_weight"], scales["hidden2_weight"])
    h2_b = qnet.tensors["hidden2_bias"].astype(np.float32) * np.float32(scales["hidden2_bias"])
    out_w = dequantize(qnet.tensors["output_weight"], scales["output_weight"])
    out_b = qnet.tensors["output_bias"].astype(np.float32) * np.float32(scales["output_bias"])

    acc = ft[np.asarray(active_features, dtype=np.int64)].sum(axis=0)
    x = np.concatenate([acc, acc])
    x = np.clip(h1_w @ x + h1_b, DEFAULT_ACTIVATION_MIN, DEFAULT_ACTIVATION_MAX)
    x = np.clip(h2_w @ x + h2_b, DEFAULT_ACTIVATION_MIN, DEFAULT_ACTIVATION_MAX)
    return float((out_w @ x + out_b)[0])
