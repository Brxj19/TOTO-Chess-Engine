#!/usr/bin/env python3
"""Extract FEN windows around large engine-eval swings in PGN comments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import chess
import chess.pgn
from tqdm import tqdm


EVAL_RE = re.compile(r"([+-]?(?:M\d+|\d+(?:\.\d+)?))/\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract blunder-window FENs from Cute Chess PGN.")
    parser.add_argument("--pgn", required=True, type=Path, help="Input PGN file.")
    parser.add_argument("--output", required=True, type=Path, help="Output FEN file.")
    parser.add_argument("--swing-threshold", type=int, default=500, help="Centipawn swing threshold.")
    parser.add_argument("--window", type=int, default=1, help="Extra plies around the swing.")
    return parser.parse_args()


def eval_from_comment(comment: str) -> int | None:
    match = EVAL_RE.search(comment)
    if not match:
        return None
    value = match.group(1)
    sign = -1 if value.startswith("-") else 1
    value = value.lstrip("+-")
    if value.startswith("M"):
        mate = int(value[1:])
        return sign * (100000 - mate)
    return int(round(float(value) * 100))


def key_for(board: chess.Board) -> str:
    ep_square = "-" if board.ep_square is None else chess.square_name(board.ep_square)
    return " ".join([board.board_fen(), "w" if board.turn else "b", board.castling_xfen(), ep_square])


def add_board(board: chess.Board, seen: set[str], fens: list[str]) -> None:
    if not board.is_valid() or board.is_checkmate() or board.is_stalemate():
        return
    key = key_for(board)
    if key not in seen:
        seen.add(key)
        fens.append(board.fen())


def process_game(game: chess.pgn.Game, threshold: int, window: int, seen: set[str], fens: list[str]) -> None:
    boards = [game.board()]
    evals: list[int | None] = [None]
    node = game

    while node.variations:
        node = node.variation(0)
        board = node.board()
        if not board.is_valid():
            return
        boards.append(board)
        evals.append(eval_from_comment(node.comment))

    for ply in range(1, len(boards)):
        prev_eval = evals[ply - 1]
        curr_eval = evals[ply]
        if prev_eval is None or curr_eval is None:
            continue
        if abs(curr_eval - prev_eval) < threshold:
            continue

        for idx in range(max(0, ply - 1 - window), min(len(boards), ply + 1 + window + 1)):
            add_board(boards[idx], seen, fens)


def main() -> None:
    args = parse_args()
    if args.swing_threshold < 1:
        raise ValueError("--swing-threshold must be positive")
    if args.window < 0:
        raise ValueError("--window must be non-negative")

    seen: set[str] = set()
    fens: list[str] = []
    games = 0

    with args.pgn.open("r", encoding="utf-8", errors="replace") as pgn_file:
        progress = tqdm(unit="game", desc="Scanning PGN")
        try:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games += 1
                process_game(game, args.swing_threshold, args.window, seen, fens)
                progress.update(1)
                progress.set_postfix(fens=len(fens))
        finally:
            progress.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for fen in fens:
            output.write(fen + "\n")

    print(f"Scanned {games} games and wrote {len(fens)} FENs to {args.output}")


if __name__ == "__main__":
    main()
