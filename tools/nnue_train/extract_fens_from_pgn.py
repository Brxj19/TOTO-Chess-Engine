#!/usr/bin/env python3
"""Extract unique legal FEN positions from a PGN file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, TextIO

import chess
import chess.pgn
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deduplicated non-terminal FEN positions from PGN games."
    )
    parser.add_argument("--pgn", required=True, type=Path, help="Input PGN file.")
    parser.add_argument(
        "--output", required=True, type=Path, help="Output text file, one FEN per line."
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of accepted games to process.",
    )
    parser.add_argument(
        "--skip-plies",
        type=int,
        default=0,
        help="Skip this many plies from the start of each accepted game.",
    )
    parser.add_argument(
        "--every-n-plies",
        type=int,
        default=1,
        help="Extract one position every N plies after --skip-plies.",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=0,
        help=(
            "Require both players to have at least this Elo. Games with missing "
            "Elo are skipped when this is greater than 0."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_games is not None and args.max_games < 1:
        raise ValueError("--max-games must be at least 1 when provided")
    if args.skip_plies < 0:
        raise ValueError("--skip-plies must be non-negative")
    if args.every_n_plies < 1:
        raise ValueError("--every-n-plies must be at least 1")
    if args.min_elo < 0:
        raise ValueError("--min-elo must be non-negative")


def header_elo(headers: chess.pgn.Headers, name: str) -> int | None:
    value = headers.get(name, "")
    try:
        return int(value)
    except ValueError:
        return None


def game_passes_elo_filter(game: chess.pgn.Game, min_elo: int) -> bool:
    if min_elo <= 0:
        return True

    white_elo = header_elo(game.headers, "WhiteElo")
    black_elo = header_elo(game.headers, "BlackElo")
    if white_elo is None or black_elo is None:
        return False

    return white_elo >= min_elo and black_elo >= min_elo


def dedupe_key(board: chess.Board) -> str:
    """Key positions by fields relevant to static evaluation and legal state."""

    ep_square = "-" if board.ep_square is None else chess.square_name(board.ep_square)
    return " ".join(
        [
            board.board_fen(),
            "w" if board.turn == chess.WHITE else "b",
            board.castling_xfen(),
            ep_square,
        ]
    )


def iter_game_fens(
    game: chess.pgn.Game,
    skip_plies: int,
    every_n_plies: int,
) -> Iterable[tuple[str, str]]:
    board = game.board()

    for ply, move in enumerate(game.mainline_moves(), start=1):
        if move not in board.legal_moves:
            break

        board.push(move)

        if ply <= skip_plies:
            continue
        if (ply - skip_plies) % every_n_plies != 0:
            continue
        if board.is_checkmate() or board.is_stalemate():
            continue

        yield dedupe_key(board), board.fen()


def extract_fens(args: argparse.Namespace) -> tuple[int, int, int, int]:
    seen: set[str] = set()
    games_read = 0
    games_processed = 0
    games_skipped_by_elo = 0
    positions_written = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.pgn.open("r", encoding="utf-8", errors="replace") as pgn_file:
        total = args.max_games if args.max_games is not None else None
        with args.output.open("w", encoding="utf-8") as output_file:
            progress = tqdm(total=total, unit="game", desc="Processing PGN")
            try:
                while args.max_games is None or games_processed < args.max_games:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    games_read += 1
                    if not game_passes_elo_filter(game, args.min_elo):
                        games_skipped_by_elo += 1
                        continue

                    written_this_game = write_unique_game_fens(
                        output_file,
                        game,
                        seen,
                        args.skip_plies,
                        args.every_n_plies,
                    )
                    positions_written += written_this_game
                    games_processed += 1
                    progress.update(1)
                    progress.set_postfix(
                        read=games_read,
                        fens=positions_written,
                        unique=len(seen),
                    )
            finally:
                progress.close()

    return games_read, games_processed, games_skipped_by_elo, positions_written


def write_unique_game_fens(
    output_file: TextIO,
    game: chess.pgn.Game,
    seen: set[str],
    skip_plies: int,
    every_n_plies: int,
) -> int:
    written = 0
    for key, fen in iter_game_fens(game, skip_plies, every_n_plies):
        if key in seen:
            continue
        seen.add(key)
        output_file.write(f"{fen}\n")
        written += 1
    return written


def main() -> None:
    args = parse_args()
    validate_args(args)

    games_read, games_processed, games_skipped_by_elo, positions_written = extract_fens(
        args
    )
    print(
        "Done: "
        f"read {games_read} games, "
        f"processed {games_processed} games, "
        f"skipped {games_skipped_by_elo} games by Elo, "
        f"wrote {positions_written} unique FENs to {args.output}"
    )


if __name__ == "__main__":
    main()
