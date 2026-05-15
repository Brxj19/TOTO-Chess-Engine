#!/usr/bin/env python3
"""Generate curated tactical/material sanity FENs for NNUE labelling."""

from __future__ import annotations

from pathlib import Path

import chess


DEFAULT_OUTPUT = Path("data/fens/tactical_material_sanity.fen")


def board_after_san(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    return board


def curated_boards() -> list[tuple[str, chess.Board]]:
    items: list[tuple[str, chess.Board]] = [
        ("hanging queen", board_after_san(["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"])),
        ("hanging rook", board_after_san(["Nf3", "d5", "Rg1", "e5", "Nxe5", "Qe7"])),
        ("hanging minor", board_after_san(["e4", "e5", "Bc4", "Nf6", "Nf3", "Nxe4"])),
        ("simple capture", board_after_san(["e4", "d5"])),
        ("knight fork", board_after_san(["e4", "e5", "Nf3", "Nc6", "Bc4", "Nd4"])),
        ("pin", board_after_san(["e4", "e5", "Nf3", "Nc6", "Bb5"])),
        ("skewer", chess.Board("4r1k1/8/8/8/8/8/4Q3/4K3 w - - 0 1")),
        ("back rank threat", chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")),
        ("queen trade", chess.Board("4k3/8/8/8/3q4/8/3Q4/4K3 w - - 0 1")),
        ("queen for rook minor", chess.Board("4k3/8/8/8/3q4/8/3R1N2/4K3 w - - 0 1")),
        ("promotion race", chess.Board("8/P6k/8/8/8/8/6p1/7K w - - 0 1")),
        ("king pawn opposition", chess.Board("8/8/8/3k4/3P4/3K4/8/8 w - - 0 1")),
        ("outside passer", chess.Board("8/8/5k2/8/5K2/6P1/7P/8 w - - 0 1")),
        ("mate in one", chess.Board("6k1/5ppp/8/8/8/8/5PPP/6KQ w - - 0 1")),
        ("mate in two style", chess.Board("6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1")),
    ]
    return items


def main() -> None:
    seen: set[str] = set()
    fens: list[str] = []

    for name, board in curated_boards():
        if not board.is_valid():
            raise ValueError(f"invalid curated board: {name}: {board.fen()}")
        if board.is_checkmate() or board.is_stalemate():
            # Training labels should come from positions where the engine can still move.
            continue
        fen = board.fen()
        key = " ".join(fen.split()[:4])
        if key not in seen:
            seen.add(key)
            fens.append(fen)

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT.open("w", encoding="utf-8") as output:
        for fen in fens:
            output.write(fen + "\n")

    print(f"Wrote {len(fens)} tactical/material FENs to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
