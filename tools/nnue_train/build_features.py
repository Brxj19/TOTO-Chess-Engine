#!/usr/bin/env python3
"""Build sparse NNUE feature datasets from labelled FEN CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import chess
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tce_nnue_train.features import (  # noqa: E402
    FEATURE_COUNT,
    debug_features_for_fen,
    extract_features,
    parse_valid_board,
)


CSV_COLUMNS = {"fen", "eval_cp"}
NPZ_SCHEMA_VERSION = 1
SAMPLE_VECTOR_PATH = SCRIPT_DIR / "test_vectors" / "features_sample.json"
SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n1p3/8/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert labelled FEN CSV rows into sparse NNUE feature arrays."
    )
    parser.add_argument("--input", type=Path, help="Input labelled CSV file.")
    parser.add_argument("--output", type=Path, help="Output compressed .npz file.")
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum CSV rows to scan. Useful for fast test builds.",
    )
    parser.add_argument(
        "--clamp-cp",
        type=int,
        default=None,
        help="Clamp eval_cp targets to +/- this value.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, keep existing rows and skip FENs already present there.",
    )
    parser.add_argument(
        "--debug-fen",
        help="Print sparse feature IDs for one FEN and exit.",
    )
    parser.add_argument(
        "--write-sample-vectors",
        action="store_true",
        help=f"Write a small feature sample JSON to {SAMPLE_VECTOR_PATH}.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.debug_fen:
        return

    if args.write_sample_vectors and not args.input and not args.output:
        return

    if args.input is None:
        raise ValueError("--input is required")
    if args.output is None:
        raise ValueError("--output is required")
    if args.max_positions is not None and args.max_positions < 1:
        raise ValueError("--max-positions must be at least 1 when provided")
    if args.clamp_cp is not None and args.clamp_cp < 1:
        raise ValueError("--clamp-cp must be at least 1 when provided")


def clamp_target(eval_cp: int, clamp_cp: int | None) -> int:
    if clamp_cp is None:
        return eval_cp
    return max(-clamp_cp, min(clamp_cp, eval_cp))


def read_required_columns(input_path: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no CSV header")
        missing = CSV_COLUMNS.difference(reader.fieldnames)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{input_path} is missing required columns: {names}")


def count_csv_rows(input_path: Path) -> int:
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return sum(1 for _ in reader)


def array_from_strings(values: list[str]) -> np.ndarray:
    return np.array(values, dtype=np.str_)


def flatten_features(feature_lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(feature_lists) + 1, dtype=np.int64)
    total = 0
    for index, features in enumerate(feature_lists):
        total += len(features)
        offsets[index + 1] = total

    flat = np.empty(total, dtype=np.int32)
    cursor = 0
    for features in feature_lists:
        end = cursor + len(features)
        flat[cursor:end] = features
        cursor = end

    return flat, offsets


def load_existing_dataset(output_path: Path) -> dict[str, np.ndarray] | None:
    if not output_path.exists():
        return None

    with np.load(output_path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def existing_fens(existing: dict[str, np.ndarray] | None) -> set[str]:
    if existing is None or "fens" not in existing:
        return set()
    return {str(fen) for fen in existing["fens"]}


def adjust_offsets(offsets: np.ndarray, base: int) -> np.ndarray:
    adjusted = offsets[1:].astype(np.int64, copy=True)
    adjusted += base
    return adjusted


def combine_with_existing(
    existing: dict[str, np.ndarray] | None,
    new_dataset: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if existing is None:
        return new_dataset

    white_base = int(existing["white_offsets"][-1])
    black_base = int(existing["black_offsets"][-1])

    return {
        "schema_version": np.array([NPZ_SCHEMA_VERSION], dtype=np.int32),
        "feature_count": np.array([FEATURE_COUNT], dtype=np.int32),
        "fens": np.concatenate([existing["fens"], new_dataset["fens"]]),
        "white_features": np.concatenate(
            [existing["white_features"], new_dataset["white_features"]]
        ),
        "white_offsets": np.concatenate(
            [
                existing["white_offsets"],
                adjust_offsets(new_dataset["white_offsets"], white_base),
            ]
        ),
        "black_features": np.concatenate(
            [existing["black_features"], new_dataset["black_features"]]
        ),
        "black_offsets": np.concatenate(
            [
                existing["black_offsets"],
                adjust_offsets(new_dataset["black_offsets"], black_base),
            ]
        ),
        "eval_cp": np.concatenate([existing["eval_cp"], new_dataset["eval_cp"]]),
        "side_to_move": np.concatenate(
            [existing["side_to_move"], new_dataset["side_to_move"]]
        ),
        "best_move": np.concatenate([existing["best_move"], new_dataset["best_move"]]),
        "depth": np.concatenate([existing["depth"], new_dataset["depth"]]),
    }


def build_npz_payload(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    white_lists = [row["white_features"] for row in rows]
    black_lists = [row["black_features"] for row in rows]
    white_features, white_offsets = flatten_features(white_lists)
    black_features, black_offsets = flatten_features(black_lists)

    return {
        "schema_version": np.array([NPZ_SCHEMA_VERSION], dtype=np.int32),
        "feature_count": np.array([FEATURE_COUNT], dtype=np.int32),
        "fens": array_from_strings([str(row["fen"]) for row in rows]),
        "white_features": white_features,
        "white_offsets": white_offsets,
        "black_features": black_features,
        "black_offsets": black_offsets,
        "eval_cp": np.array([int(row["eval_cp"]) for row in rows], dtype=np.int32),
        "side_to_move": np.array(
            [int(row["side_to_move"]) for row in rows], dtype=np.uint8
        ),
        "best_move": array_from_strings([str(row["best_move"]) for row in rows]),
        "depth": np.array([int(row["depth"]) for row in rows], dtype=np.int16),
    }


def iter_label_rows(input_path: Path) -> Iterable[dict[str, str]]:
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            yield row


def build_features(args: argparse.Namespace) -> tuple[int, int, int, int]:
    read_required_columns(args.input)

    existing = load_existing_dataset(args.output) if args.resume else None
    seen_fens = existing_fens(existing)
    total_rows = count_csv_rows(args.input)
    total = (
        min(args.max_positions, total_rows)
        if args.max_positions is not None
        else total_rows
    )

    rows: list[dict[str, object]] = []
    scanned = 0
    skipped_invalid = 0
    skipped_resume = 0

    progress = tqdm(total=total, unit="row", desc="Building features")
    try:
        for row in iter_label_rows(args.input):
            if args.max_positions is not None and scanned >= args.max_positions:
                break

            scanned += 1
            fen = row.get("fen", "").strip()
            if not fen:
                skipped_invalid += 1
                progress.update(1)
                continue

            if fen in seen_fens:
                skipped_resume += 1
                progress.update(1)
                continue

            try:
                board = parse_valid_board(fen)
                eval_cp = clamp_target(int(row["eval_cp"]), args.clamp_cp)
                depth = int(row["depth"]) if row.get("depth") else -1
                sparse = extract_features(board)
            except (ValueError, KeyError) as exc:
                skipped_invalid += 1
                tqdm.write(f"warning: skipped row for FEN {fen}: {exc}")
                progress.update(1)
                continue

            rows.append(
                {
                    "fen": fen,
                    "white_features": sparse.white,
                    "black_features": sparse.black,
                    "eval_cp": eval_cp,
                    "side_to_move": 0 if board.turn == chess.WHITE else 1,
                    "best_move": (row.get("best_move") or "").strip(),
                    "depth": depth,
                }
            )
            seen_fens.add(fen)
            progress.update(1)
            progress.set_postfix(
                kept=len(rows),
                invalid=skipped_invalid,
                resume=skipped_resume,
            )
    finally:
        progress.close()

    new_dataset = build_npz_payload(rows)
    dataset = combine_with_existing(existing, new_dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **dataset)

    return scanned, len(rows), skipped_invalid, skipped_resume


def write_sample_vectors(path: Path = SAMPLE_VECTOR_PATH) -> None:
    vectors = [debug_features_for_fen(fen) for fen in SAMPLE_FENS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(vectors, json_file, indent=2)
        json_file.write("\n")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.debug_fen:
        print(json.dumps(debug_features_for_fen(args.debug_fen), indent=2))
        return

    if args.write_sample_vectors and not args.input and not args.output:
        write_sample_vectors()
        print(f"Wrote sample vectors to {SAMPLE_VECTOR_PATH}")
        return

    if args.write_sample_vectors:
        write_sample_vectors()

    scanned, kept, skipped_invalid, skipped_resume = build_features(args)
    print(
        "Done: "
        f"scanned {scanned} rows, "
        f"built {kept} new positions, "
        f"skipped {skipped_invalid} invalid rows, "
        f"skipped {skipped_resume} existing rows, "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
