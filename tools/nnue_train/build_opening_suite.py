#!/usr/bin/env python3
"""Build an EPD opening suite for varied engine testing."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess.pgn
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract opening EPDs from PGN games.")
    parser.add_argument("--pgn", required=True, type=Path, help="Input PGN file.")
    parser.add_argument("--output", required=True, type=Path, help="Output EPD file.")
    parser.add_argument("--max-games", type=int, default=None, help="Maximum accepted games.")
    parser.add_argument("--min-elo", type=int, default=1600, help="Minimum Elo for both players.")
    parser.add_argument("--min-ply", type=int, default=8, help="First ply to consider.")
    parser.add_argument("--max-ply", type=int, default=24, help="Last ply to consider.")
    parser.add_argument("--every-n-plies", type=int, default=4, help="Sampling interval.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum EPD positions to write.")
    parser.add_argument("--seed", type=int, default=1, help="Shuffle seed.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_games is not None and args.max_games < 1:
        raise ValueError("--max-games must be at least 1")
    if args.min_elo < 0:
        raise ValueError("--min-elo must be non-negative")
    if args.min_ply < 0 or args.max_ply < args.min_ply:
        raise ValueError("--min-ply/--max-ply are invalid")
    if args.every_n_plies < 1:
        raise ValueError("--every-n-plies must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")


def header_elo(game: chess.pgn.Game, name: str) -> int | None:
    try:
        return int(game.headers.get(name, ""))
    except ValueError:
        return None


def passes_elo(game: chess.pgn.Game, min_elo: int) -> bool:
    white_elo = header_elo(game, "WhiteElo")
    black_elo = header_elo(game, "BlackElo")
    return white_elo is not None and black_elo is not None and white_elo >= min_elo and black_elo >= min_elo


def epd_key(board: chess.Board) -> str:
    ep_square = "-" if board.ep_square is None else chess.square_name(board.ep_square)
    return " ".join([board.board_fen(), "w" if board.turn else "b", board.castling_xfen(), ep_square])


def main() -> None:
    args = parse_args()
    validate_args(args)

    seen: set[str] = set()
    epds: list[str] = []
    accepted_games = 0
    read_games = 0

    with args.pgn.open("r", encoding="utf-8", errors="replace") as pgn_file:
        progress = tqdm(total=args.max_games, unit="game", desc="Reading PGN")
        try:
            while args.max_games is None or accepted_games < args.max_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                read_games += 1
                if not passes_elo(game, args.min_elo):
                    continue

                board = game.board()
                valid_game = True
                for ply, move in enumerate(game.mainline_moves(), start=1):
                    if move not in board.legal_moves:
                        valid_game = False
                        break
                    board.push(move)
                    if ply < args.min_ply or ply > args.max_ply:
                        continue
                    if (ply - args.min_ply) % args.every_n_plies != 0:
                        continue
                    if board.is_checkmate() or board.is_stalemate():
                        continue
                    key = epd_key(board)
                    if key not in seen:
                        seen.add(key)
                        epds.append(board.epd())

                if valid_game:
                    accepted_games += 1
                    progress.update(1)
                    progress.set_postfix(read=read_games, epds=len(epds))
        finally:
            progress.close()

    random.Random(args.seed).shuffle(epds)
    if args.limit is not None:
        epds = epds[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write("# TCE opening suite for engine testing variety.\n")
        output.write("# Not a training dataset. Use with Cute Chess opening suite settings.\n")
        for epd in epds:
            output.write(epd + "\n")

    print(f"Wrote {len(epds)} EPD positions to {args.output}")


if __name__ == "__main__":
    main()
