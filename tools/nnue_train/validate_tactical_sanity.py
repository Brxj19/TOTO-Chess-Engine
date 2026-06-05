#!/usr/bin/env python3
"""Validate a trained TCE NNUE checkpoint on tactical/material sanity positions."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import chess
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tce_nnue_train.dataset import collate_feature_batch  # noqa: E402
from tce_nnue_train.features import FEATURE_COUNT, extract_features, parse_valid_board  # noqa: E402
from tce_nnue_train.model import TceNnueModel  # noqa: E402
from tce_nnue_train.train import forward_model, move_batch_to_device, resolve_device  # noqa: E402


CRITICAL_CATEGORIES = {
    "queen_up",
    "queen_down",
    "rook_up",
    "rook_down",
    "minor_up",
    "queen_vs_rook_minor",
    "simple_capture",
    "fork",
    "promotion_race",
    "mate_in_1",
    "mate_threat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tactical/material sanity positions.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Input trained .pt checkpoint.")
    parser.add_argument("--labels", required=True, type=Path, help="Labelled sanity CSV.")
    parser.add_argument("--metadata", type=Path, help="Optional FEN metadata CSV.")
    parser.add_argument("--output", type=Path, help="Optional per-position CSV output.")
    parser.add_argument("--target-scale", type=float, default=None, help="Override checkpoint target scale.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--material-threshold", type=int, default=500)
    parser.add_argument("--min-pred-cp", type=int, default=200)
    parser.add_argument(
        "--report-cap-cp",
        type=int,
        default=3000,
        help="Cap target/prediction values for MAE and pass/fail reporting; raw values stay in CSV.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a valid TCE NNUE checkpoint")
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


def model_config(checkpoint: dict[str, Any]) -> dict[str, int]:
    config = dict(checkpoint.get("model_config", {}))
    config.setdefault("feature_count", FEATURE_COUNT)
    config.setdefault("half_dim", 128)
    config.setdefault("hidden1_dim", 64)
    config.setdefault("hidden2_dim", 32)
    return config


def load_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        for row in reader:
            fen = (row.get("fen") or "").strip()
            if fen:
                result[fen] = {
                    "category": (row.get("category") or "unknown").strip() or "unknown",
                    "description": (row.get("description") or "").strip(),
                }
    return result


def iter_label_rows(path: Path, max_rows: int | None):
    with path.open("r", encoding="utf-8", newline="") as label_file:
        reader = csv.DictReader(label_file)
        count = 0
        for row in reader:
            if max_rows is not None and count >= max_rows:
                break
            count += 1
            yield row


def row_to_sample(row: dict[str, str], target_scale: float) -> dict[str, Any]:
    fen = (row.get("fen") or "").strip()
    board = parse_valid_board(fen)
    sparse = extract_features(board)
    eval_cp = float(row["eval_cp"])
    return {
        "white_features": torch.tensor(sparse.white, dtype=torch.long),
        "black_features": torch.tensor(sparse.black, dtype=torch.long),
        "side_to_move": 0 if board.turn == chess.WHITE else 1,
        "target": max(-2.0, min(2.0, eval_cp / target_scale)),
        "eval_cp": eval_cp,
        "best_move": (row.get("best_move") or "").strip(),
        "depth": int(row["depth"]) if row.get("depth") else -1,
        "fen": fen,
    }


def same_sign(a: float, b: float) -> bool:
    if a == 0 or b == 0:
        return abs(a - b) < 50.0
    return (a > 0) == (b > 0)


def clamp_cp(value: float, cap: int) -> float:
    return max(float(-cap), min(float(cap), value))


def pass_material(row: dict[str, Any], material_threshold: int, min_pred_cp: int) -> bool:
    if abs(row["report_target_cp"]) < material_threshold:
        return True
    return row["sign_correct"] and abs(row["report_pred_cp"]) >= min_pred_cp


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.material_threshold < 1:
        raise ValueError("--material-threshold must be positive")
    if args.min_pred_cp < 0:
        raise ValueError("--min-pred-cp must be non-negative")
    if args.report_cap_cp < args.material_threshold:
        raise ValueError("--report-cap-cp must be at least --material-threshold")

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    target_scale = checkpoint_target_scale(checkpoint, args.target_scale)
    metadata = load_metadata(args.metadata)

    model = TceNnueModel(**model_config(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples: list[dict[str, Any]] = []
    skipped = 0
    for label_row in iter_label_rows(args.labels, args.max_rows):
        try:
            sample = row_to_sample(label_row, target_scale)
        except (ValueError, KeyError):
            skipped += 1
            continue
        samples.append(sample)

    if not samples:
        raise ValueError("no valid labelled sanity positions found")

    batch = move_batch_to_device(collate_feature_batch(samples), device)
    pred_cp = (forward_model(model, batch).cpu() * target_scale).tolist()
    target_cp = batch.eval_cp.cpu().tolist()

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        meta = metadata.get(sample["fen"], {})
        category = meta.get("category", "unknown")
        target = float(target_cp[index])
        pred = float(pred_cp[index])
        report_target = clamp_cp(target, args.report_cap_cp)
        report_pred = clamp_cp(pred, args.report_cap_cp)
        result = {
            "index": index,
            "category": category,
            "description": meta.get("description", ""),
            "fen": sample["fen"],
            "target_cp": target,
            "pred_cp": pred,
            "report_target_cp": report_target,
            "report_pred_cp": report_pred,
            "abs_error_cp": abs(report_pred - report_target),
            "sign_correct": same_sign(target, pred),
            "best_move": sample["best_move"],
            "depth": sample["depth"],
        }
        result["material_pass"] = pass_material(result, args.material_threshold, args.min_pred_cp)
        result["critical"] = category in CRITICAL_CATEGORIES
        rows.append(result)

    total = len(rows)
    mae = sum(row["abs_error_cp"] for row in rows) / total
    sign_rate = sum(1 for row in rows if row["sign_correct"]) / total
    material_rows = [row for row in rows if abs(row["report_target_cp"]) >= args.material_threshold]
    material_pass_rate = (
        sum(1 for row in material_rows if row["material_pass"]) / len(material_rows)
        if material_rows else 1.0
    )
    critical_rows = [row for row in rows if row["critical"]]
    critical_pass_rate = (
        sum(1 for row in critical_rows if row["material_pass"]) / len(critical_rows)
        if critical_rows else 1.0
    )

    print(f"checkpoint:            {args.checkpoint}")
    print(f"labels:                {args.labels}")
    print(f"device:                {device}")
    print(f"positions:             {total}")
    print(f"skipped_invalid:       {skipped}")
    print(f"report_cap_cp:         {args.report_cap_cp}")
    print(f"mae_cp:                {mae:.2f}")
    print(f"sign_correct_rate:     {sign_rate:.3f}")
    print(f"material_pass_rate:    {material_pass_rate:.3f}")
    print(f"critical_pass_rate:    {critical_pass_rate:.3f}")
    print()
    print("category\tcount\tmae_cp\tsign_rate\tmaterial_pass")

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    for category in sorted(by_category):
        group = by_category[category]
        group_mae = sum(row["abs_error_cp"] for row in group) / len(group)
        group_sign = sum(1 for row in group if row["sign_correct"]) / len(group)
        group_pass = sum(1 for row in group if row["material_pass"]) / len(group)
        print(f"{category}\t{len(group)}\t{group_mae:.1f}\t{group_sign:.3f}\t{group_pass:.3f}")

    print()
    print("worst_positions:")
    for row in sorted(rows, key=lambda item: item["abs_error_cp"], reverse=True)[:12]:
        print(
            f"{row['category']}\t"
            f"target={row['target_cp']:.0f}\t"
            f"pred={row['pred_cp']:.0f}\t"
            f"report_target={row['report_target_cp']:.0f}\t"
            f"report_pred={row['report_pred_cp']:.0f}\t"
            f"err={row['abs_error_cp']:.0f}\t"
            f"sign={int(row['sign_correct'])}\t"
            f"pass={int(row['material_pass'])}\t"
            f"{row['fen']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
