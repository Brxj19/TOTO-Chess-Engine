#!/usr/bin/env python3
"""Label FEN positions with a UCI engine evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TextIO

import chess
import chess.engine
from tqdm import tqdm


CSV_COLUMNS = ["fen", "eval_cp", "best_move", "depth"]
MATE_SCORE = 100000
FLUSH_EVERY = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FEN positions with a UCI chess engine."
    )
    parser.add_argument(
        "--engine-path",
        required=True,
        help="Path or command name for the UCI engine, for example 'stockfish'.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .fen file.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV file.")
    parser.add_argument(
        "--depth", required=True, type=int, help="Engine search depth per position."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of input FEN rows to scan.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Engine Threads option, when supported.",
    )
    parser.add_argument(
        "--hash",
        type=int,
        default=None,
        help="Engine Hash option in MB, when supported.",
    )
    parser.add_argument(
        "--clamp-cp",
        type=int,
        default=3000,
        help="Clamp non-mate centipawn evaluations to +/- this value.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip FENs already present in the output CSV.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.depth < 1:
        raise ValueError("--depth must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided")
    if args.threads is not None and args.threads < 1:
        raise ValueError("--threads must be at least 1 when provided")
    if args.hash is not None and args.hash < 1:
        raise ValueError("--hash must be at least 1 when provided")
    if args.clamp_cp < 1:
        raise ValueError("--clamp-cp must be at least 1")


def load_labelled_fens(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    labelled: set[str] = set()
    with output_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return labelled
        if "fen" not in reader.fieldnames:
            raise ValueError(f"{output_path} does not contain a 'fen' column")

        for row in reader:
            fen = row.get("fen", "").strip()
            if fen:
                labelled.add(fen)

    return labelled


def count_input_fens(input_path: Path) -> int:
    count = 0
    with input_path.open("r", encoding="utf-8", errors="replace") as fen_file:
        for line in fen_file:
            if line.strip():
                count += 1
    return count


def configure_engine(engine: chess.engine.SimpleEngine, args: argparse.Namespace) -> None:
    options: dict[str, int] = {}
    if args.threads is not None:
        options["Threads"] = args.threads
    if args.hash is not None:
        options["Hash"] = args.hash

    for name, value in options.items():
        try:
            engine.configure({name: value})
        except chess.engine.EngineError as exc:
            tqdm.write(f"warning: engine does not accept {name}={value}: {exc}")


def score_to_cp(score: chess.engine.PovScore, board: chess.Board, clamp_cp: int) -> int:
    relative_score = score.pov(board.turn)
    mate = relative_score.mate()
    if mate is not None:
        if mate > 0:
            return MATE_SCORE - mate
        return -MATE_SCORE + abs(mate)

    cp = relative_score.score()
    if cp is None:
        raise ValueError("engine returned neither centipawn nor mate score")

    return max(-clamp_cp, min(clamp_cp, cp))


def analyse_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
    clamp_cp: int,
) -> tuple[int, str]:
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    if "score" not in info:
        raise ValueError("engine analysis did not include a score")

    eval_cp = score_to_cp(info["score"], board, clamp_cp)
    pv = info.get("pv", [])
    best_move = pv[0].uci() if pv else ""
    return eval_cp, best_move


def open_output_csv(output_path: Path, resume: bool) -> tuple[TextIO, csv.DictWriter]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    append = resume and output_path.exists() and output_path.stat().st_size > 0
    csv_file = output_path.open("a" if append else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if not append:
        writer.writeheader()
        csv_file.flush()
    return csv_file, writer


def label_positions(args: argparse.Namespace) -> tuple[int, int, int, int]:
    labelled_fens = load_labelled_fens(args.output) if args.resume else set()
    total_input = count_input_fens(args.input)
    total = min(args.limit, total_input) if args.limit is not None else total_input

    labelled = 0
    scanned = 0
    skipped_resume = 0
    skipped_invalid = 0
    failed = 0

    engine = chess.engine.SimpleEngine.popen_uci(args.engine_path)
    try:
        configure_engine(engine, args)

        with args.input.open("r", encoding="utf-8", errors="replace") as fen_file:
            output_file, writer = open_output_csv(args.output, args.resume)
            with output_file:
                progress = tqdm(total=total, unit="fen", desc="Labelling FENs")
                try:
                    for line in fen_file:
                        if args.limit is not None and scanned >= args.limit:
                            break

                        fen = line.strip()
                        if not fen:
                            continue

                        scanned += 1

                        if args.resume and fen in labelled_fens:
                            skipped_resume += 1
                            progress.update(1)
                            continue

                        try:
                            board = chess.Board(fen)
                        except ValueError as exc:
                            skipped_invalid += 1
                            tqdm.write(f"warning: invalid FEN skipped: {fen} ({exc})")
                            progress.update(1)
                            continue

                        if not board.is_valid():
                            skipped_invalid += 1
                            tqdm.write(
                                "warning: invalid chess position skipped: "
                                f"{fen} (status={board.status()})"
                            )
                            progress.update(1)
                            continue

                        try:
                            eval_cp, best_move = analyse_position(
                                engine,
                                board,
                                args.depth,
                                args.clamp_cp,
                            )
                        except (
                            chess.engine.EngineError,
                            chess.engine.EngineTerminatedError,
                            chess.engine.EngineTimeoutError,
                            ValueError,
                        ) as exc:
                            failed += 1
                            tqdm.write(f"warning: failed to label FEN: {fen} ({exc})")
                            progress.update(1)
                            continue

                        writer.writerow(
                            {
                                "fen": fen,
                                "eval_cp": eval_cp,
                                "best_move": best_move,
                                "depth": args.depth,
                            }
                        )
                        labelled_fens.add(fen)
                        labelled += 1

                        if labelled % FLUSH_EVERY == 0:
                            output_file.flush()

                        progress.update(1)
                        progress.set_postfix(
                            labelled=labelled,
                            resume=skipped_resume,
                            invalid=skipped_invalid,
                        )
                finally:
                    output_file.flush()
                    progress.close()
    finally:
        engine.quit()

    return labelled, skipped_resume, skipped_invalid, failed


def main() -> None:
    args = parse_args()
    validate_args(args)

    labelled, skipped_resume, skipped_invalid, failed = label_positions(args)
    print(
        "Done: "
        f"labelled {labelled} positions, "
        f"skipped {skipped_resume} already-labelled positions, "
        f"skipped {skipped_invalid} invalid FENs, "
        f"failed {failed} engine analyses, "
        f"wrote labels to {args.output}"
    )


if __name__ == "__main__":
    main()
