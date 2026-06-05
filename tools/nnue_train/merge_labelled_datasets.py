#!/usr/bin/env python3
"""Merge labelled NNUE CSV datasets with optional tactical oversampling."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import chess


EXPECTED_COLUMNS = ["fen", "eval_cp", "best_move", "depth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge labelled FEN CSV datasets.")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path, help="Input labelled CSV files.")
    parser.add_argument("--output", required=True, type=Path, help="Merged output CSV.")
    parser.add_argument("--seed", type=int, default=1, help="Shuffle seed.")
    parser.add_argument(
        "--oversample",
        action="append",
        default=[],
        help="Oversample FILE:FACTOR, for example data/labels/tactical.csv:5.",
    )
    parser.add_argument(
        "--add-mirrors",
        action="store_true",
        help="Add color-swapped mirrored FEN rows to balance side-to-move coverage.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = set(EXPECTED_COLUMNS).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        return [{name: (row.get(name) or "") for name in EXPECTED_COLUMNS} for row in reader]


def parse_oversamples(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"invalid --oversample value: {value}")
        path_text, factor_text = value.rsplit(":", 1)
        factor = int(factor_text)
        if factor < 1:
            raise ValueError("oversample factor must be at least 1")
        result[str(Path(path_text))] = factor
    return result


def mirror_uci_move(move_text: str) -> str:
    if not move_text:
        return ""
    try:
        move = chess.Move.from_uci(move_text)
    except ValueError:
        return ""
    mirrored = chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )
    return mirrored.uci()


def mirrored_row(row: dict[str, str]) -> dict[str, str] | None:
    try:
        board = chess.Board(row["fen"])
    except ValueError:
        return None
    if not board.is_valid():
        return None

    mirrored = board.mirror()
    if not mirrored.is_valid():
        return None

    return {
        "fen": mirrored.fen(),
        "eval_cp": row["eval_cp"],
        "best_move": mirror_uci_move(row.get("best_move", "")),
        "depth": row["depth"],
    }


def main() -> None:
    args = parse_args()
    oversamples = parse_oversamples(args.oversample)
    rows: list[dict[str, str]] = []
    extra_rows: list[dict[str, str]] = []

    for path in args.inputs:
        factor = oversamples.get(str(path), 1)
        input_rows = read_rows(path)
        rows.extend(input_rows)
        if factor > 1:
            extra_rows.extend(input_rows * (factor - 1))

    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        fen = row["fen"].strip()
        if not fen or fen in seen:
            continue
        seen.add(fen)
        row["fen"] = fen
        deduped.append(row)

    # Dedupe first, then intentionally repeat selected tactical rows for training emphasis.
    extras = [row for row in extra_rows if row["fen"].strip() in seen]
    deduped.extend(extras)

    if args.add_mirrors:
        mirrored_rows: list[dict[str, str]] = []
        for row in list(deduped):
            mirror = mirrored_row(row)
            if mirror is not None:
                mirrored_rows.append(mirror)
        deduped.extend(mirrored_rows)

    random.Random(args.seed).shuffle(deduped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Wrote {len(deduped)} rows to {args.output}")


if __name__ == "__main__":
    main()
