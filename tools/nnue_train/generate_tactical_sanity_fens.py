#!/usr/bin/env python3
"""Generate curated and synthetic tactical/material sanity FENs for NNUE labelling."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import chess


DEFAULT_OUTPUT = Path("data/fens/tactical_material_sanity.fen")
DEFAULT_METADATA_OUTPUT = Path("data/fens/tactical_material_sanity_meta.csv")


@dataclass(frozen=True)
class SanityPosition:
    category: str
    description: str
    board: chess.Board


FILES = list(range(8))
LIGHT_PIECES = [chess.KNIGHT, chess.BISHOP]


def board_after_san(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    return board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TCE NNUE tactical/material sanity FENs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    return parser.parse_args()


def curated_positions() -> list[SanityPosition]:
    items: list[SanityPosition] = [
        SanityPosition("hanging_queen", "early queen exposed to Nf6", board_after_san(["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"])),
        SanityPosition("hanging_rook", "rook lift can be punished", board_after_san(["Nf3", "d5", "Rg1", "e5", "Nxe5", "Qe7"])),
        SanityPosition("hanging_minor", "minor piece exposed in open e-file", board_after_san(["e4", "e5", "Bc4", "Nf6", "Nf3", "Nxe4"])),
        SanityPosition("simple_capture", "center pawn capture available", board_after_san(["e4", "d5"])),
        SanityPosition("fork", "knight fork motif", board_after_san(["e4", "e5", "Nf3", "Nc6", "Bc4", "Nd4"])),
        SanityPosition("pin", "bishop pin on knight", board_after_san(["e4", "e5", "Nf3", "Nc6", "Bb5"])),
        SanityPosition("skewer", "queen skewers rook and king", chess.Board("4r1k1/8/8/8/8/8/4Q3/4K3 w - - 0 1")),
        SanityPosition("back_rank", "rook pressure on back rank", chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")),
        SanityPosition("queen_trade", "equal queens facing", chess.Board("4k3/8/8/8/3q4/8/3Q4/4K3 w - - 0 1")),
        SanityPosition("queen_for_rook_minor", "queen versus rook and knight imbalance", chess.Board("4k3/8/8/8/3q4/8/3R1N2/4K3 w - - 0 1")),
        SanityPosition("promotion_race", "opposite passed pawns", chess.Board("8/P6k/8/8/8/8/6p1/7K w - - 0 1")),
        SanityPosition("pawn_endgame", "king pawn opposition", chess.Board("8/8/8/3k4/3P4/3K4/8/8 w - - 0 1")),
        SanityPosition("pawn_endgame", "outside passer", chess.Board("8/8/5k2/8/5K2/6P1/7P/8 w - - 0 1")),
        SanityPosition("mate_in_1", "queen mate in one", chess.Board("6k1/5ppp/8/8/8/8/5PPP/6KQ w - - 0 1")),
        SanityPosition("mate_threat", "rook mate net", chess.Board("6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1")),
    ]
    return items


def empty_board(turn: chess.Color) -> chess.Board:
    board = chess.Board(None)
    board.turn = turn
    board.clear_stack()
    board.castling_rights = 0
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1
    return board


def make_board(turn: chess.Color, pieces: list[tuple[chess.PieceType, chess.Color, chess.Square]]) -> chess.Board:
    board = empty_board(turn)
    for piece_type, color, square in pieces:
        board.set_piece_at(square, chess.Piece(piece_type, color))
    return board


def add_if_valid(items: list[SanityPosition], category: str, description: str, board: chess.Board) -> None:
    if board.is_valid() and not board.is_checkmate() and not board.is_stalemate():
        items.append(SanityPosition(category, description, board))


def synthetic_material_positions() -> list[SanityPosition]:
    items: list[SanityPosition] = []

    for turn in [chess.WHITE, chess.BLACK]:
        stm = "white" if turn == chess.WHITE else "black"
        for file_index, file_ in enumerate(FILES):
            wk = chess.square(4, 0)
            bk = chess.square(4, 7)
            own_rank = 2 if turn == chess.WHITE else 5
            opp_rank = 5 if turn == chess.WHITE else 2
            own_color = turn
            opp_color = not turn

            qsq = chess.square(file_, own_rank)
            rsq = chess.square((file_index + 2) % 8, own_rank)
            bsq = chess.square((file_index + 4) % 8, own_rank)
            nsq = chess.square((file_index + 5) % 8, own_rank)
            opp_qsq = chess.square(file_, opp_rank)
            opp_rsq = chess.square((file_index + 2) % 8, opp_rank)

            add_if_valid(
                items,
                "queen_up",
                f"{stm} has extra queen file {file_index}",
                make_board(turn, [(chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk), (chess.QUEEN, own_color, qsq)]),
            )
            add_if_valid(
                items,
                "queen_down",
                f"{stm} faces extra queen file {file_index}",
                make_board(turn, [(chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk), (chess.QUEEN, opp_color, opp_qsq)]),
            )
            add_if_valid(
                items,
                "rook_up",
                f"{stm} has extra rook file {file_index}",
                make_board(turn, [(chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk), (chess.ROOK, own_color, rsq)]),
            )
            add_if_valid(
                items,
                "rook_down",
                f"{stm} faces extra rook file {file_index}",
                make_board(turn, [(chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk), (chess.ROOK, opp_color, opp_rsq)]),
            )
            add_if_valid(
                items,
                "minor_up",
                f"{stm} has bishop and knight file {file_index}",
                make_board(turn, [(chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk), (chess.BISHOP, own_color, bsq), (chess.KNIGHT, own_color, nsq)]),
            )
            add_if_valid(
                items,
                "queen_vs_rook_minor",
                f"{stm} queen versus rook minor file {file_index}",
                make_board(turn, [
                    (chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk),
                    (chess.QUEEN, own_color, qsq), (chess.ROOK, opp_color, opp_rsq),
                    (LIGHT_PIECES[file_index % 2], opp_color, chess.square((file_index + 3) % 8, opp_rank)),
                ]),
            )
            add_if_valid(
                items,
                "pawn_endgame",
                f"{stm} two-pawn king ending file {file_index}",
                make_board(turn, [
                    (chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk),
                    (chess.PAWN, own_color, chess.square(file_index, own_rank)),
                    (chess.PAWN, own_color, chess.square((file_index + 1) % 8, own_rank)),
                    (chess.PAWN, opp_color, chess.square((file_index + 4) % 8, opp_rank)),
                ]),
            )
            add_if_valid(
                items,
                "hanging_queen",
                f"{stm} queen and rook imbalance file {file_index}",
                make_board(turn, [
                    (chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk),
                    (chess.QUEEN, opp_color, opp_qsq),
                    (chess.ROOK, own_color, rsq),
                    (chess.BISHOP, own_color, bsq),
                ]),
            )

    return items


def synthetic_tactical_positions() -> list[SanityPosition]:
    items: list[SanityPosition] = []

    for turn in [chess.WHITE, chess.BLACK]:
        own_color = turn
        opp_color = not turn
        stm = "white" if turn == chess.WHITE else "black"
        wk = chess.square(6, 0) if turn == chess.WHITE else chess.square(6, 7)
        bk = chess.square(6, 7) if turn == chess.WHITE else chess.square(6, 0)
        own_back = 0 if turn == chess.WHITE else 7
        opp_back = 7 if turn == chess.WHITE else 0
        own_mid = 3 if turn == chess.WHITE else 4
        opp_mid = 4 if turn == chess.WHITE else 3
        pawn_rank = 6 if turn == chess.WHITE else 1

        for file_index in range(8):
            center_file = (file_index + 2) % 8
            add_if_valid(
                items,
                "simple_capture",
                f"{stm} can win loose queen file {file_index}",
                make_board(turn, [
                    (chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk),
                    (chess.ROOK, own_color, chess.square(center_file, own_mid)),
                    (chess.QUEEN, opp_color, chess.square(center_file, opp_mid)),
                ]),
            )
            add_if_valid(
                items,
                "fork",
                f"{stm} knight attacks queen and rook file {file_index}",
                make_board(turn, [
                    (chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk),
                    (chess.KNIGHT, own_color, chess.square(center_file, own_mid)),
                    (chess.QUEEN, opp_color, chess.square((center_file + 1) % 8, opp_mid)),
                    (chess.ROOK, opp_color, chess.square((center_file + 7) % 8, opp_mid)),
                ]),
            )
            add_if_valid(
                items,
                "pin",
                f"{stm} pins queen to king file {file_index}",
                make_board(turn, [
                    (chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk),
                    (chess.BISHOP, own_color, chess.square(file_index, own_mid)),
                    (chess.QUEEN, opp_color, chess.square((file_index + 1) % 8, opp_mid)),
                ]),
            )
            add_if_valid(
                items,
                "promotion_race",
                f"{stm} advanced passer file {file_index}",
                make_board(turn, [
                    (chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk),
                    (chess.PAWN, own_color, chess.square(file_index, pawn_rank)),
                    (chess.PAWN, opp_color, chess.square((file_index + 4) % 8, 1 if turn == chess.WHITE else 6)),
                ]),
            )
            add_if_valid(
                items,
                "back_rank",
                f"{stm} back-rank pressure file {file_index}",
                make_board(turn, [
                    (chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk),
                    (chess.ROOK, own_color, chess.square(file_index, opp_back)),
                    (chess.PAWN, opp_color, chess.square((file_index + 1) % 8, opp_mid)),
                    (chess.PAWN, opp_color, chess.square((file_index + 2) % 8, opp_mid)),
                ]),
            )
            add_if_valid(
                items,
                "queen_trade",
                f"{stm} queen trade file {file_index}",
                make_board(turn, [
                    (chess.KING, chess.WHITE, wk), (chess.KING, chess.BLACK, bk),
                    (chess.QUEEN, own_color, chess.square(file_index, own_mid)),
                    (chess.QUEEN, opp_color, chess.square(file_index, opp_mid)),
                ]),
            )

    return items


def all_positions() -> list[SanityPosition]:
    return curated_positions() + synthetic_material_positions() + synthetic_tactical_positions()


def dedupe_positions(positions: list[SanityPosition]) -> list[SanityPosition]:
    seen: set[str] = set()
    result: list[SanityPosition] = []
    for item in positions:
        if not item.board.is_valid():
            raise ValueError(f"invalid generated board: {item.category}: {item.board.fen()}")
        if item.board.is_checkmate() or item.board.is_stalemate():
            continue
        key = " ".join(item.board.fen().split()[:4])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def main() -> None:
    args = parse_args()
    positions = dedupe_positions(all_positions())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for item in positions:
            fen = item.board.fen()
            output.write(fen + "\n")

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_output.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=["fen", "category", "description"])
        writer.writeheader()
        for item in positions:
            writer.writerow({
                "fen": item.board.fen(),
                "category": item.category,
                "description": item.description,
            })

    print(f"Wrote {len(positions)} tactical/material FENs to {args.output}")
    print(f"Wrote metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
