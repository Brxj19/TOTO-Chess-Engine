"""Sparse feature extraction for the first TCE-owned NNUE dataset.

The initial feature set is intentionally simple:

    feature = king_square * (12 * 64) + piece_type * 64 + piece_square

`king_square` is the perspective side's king square after orientation.
`piece_type` is perspective-relative:

    0..5  = own pawn, knight, bishop, rook, queen, king
    6..11 = opponent pawn, knight, bishop, rook, queen, king

Kings are used as the bucket selector and are not emitted as normal piece
features by default. The resulting feature space has 64 * 12 * 64 = 49152
possible sparse IDs per perspective.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess


KING_SQUARES = 64
PIECE_TYPES = 12
PIECE_SQUARES = 64
FEATURE_COUNT = KING_SQUARES * PIECE_TYPES * PIECE_SQUARES

OWN_PIECE_BASE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
OPPONENT_OFFSET = 6


@dataclass(frozen=True)
class PositionFeatures:
    white: list[int]
    black: list[int]


def orient_square(square: chess.Square, perspective: chess.Color) -> int:
    """Return a square index in the perspective-relative board orientation."""

    if perspective == chess.WHITE:
        return square
    return square ^ 63


def perspective_piece_type(piece: chess.Piece, perspective: chess.Color) -> int:
    piece_type = OWN_PIECE_BASE[piece.piece_type]
    if piece.color != perspective:
        piece_type += OPPONENT_OFFSET
    return piece_type


def feature_id(king_square: int, piece_type: int, piece_square: int) -> int:
    return (king_square * PIECE_TYPES + piece_type) * PIECE_SQUARES + piece_square


def king_square_for(board: chess.Board, perspective: chess.Color) -> int:
    king_square = board.king(perspective)
    if king_square is None:
        color_name = "white" if perspective == chess.WHITE else "black"
        raise ValueError(f"position has no {color_name} king")
    return orient_square(king_square, perspective)


def extract_perspective_features(
    board: chess.Board,
    perspective: chess.Color,
    include_kings: bool = False,
) -> list[int]:
    """Extract sorted sparse feature IDs for one perspective."""

    king_square = king_square_for(board, perspective)
    features: list[int] = []

    for square, piece in board.piece_map().items():
        if not include_kings and piece.piece_type == chess.KING:
            continue

        oriented_square = orient_square(square, perspective)
        piece_type = perspective_piece_type(piece, perspective)
        features.append(feature_id(king_square, piece_type, oriented_square))

    features.sort()
    return features


def extract_features(board: chess.Board, include_kings: bool = False) -> PositionFeatures:
    return PositionFeatures(
        white=extract_perspective_features(board, chess.WHITE, include_kings),
        black=extract_perspective_features(board, chess.BLACK, include_kings),
    )


def parse_valid_board(fen: str) -> chess.Board:
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"invalid chess position: status={board.status()}")
    return board


def debug_features_for_fen(fen: str) -> dict[str, object]:
    board = parse_valid_board(fen)
    features = extract_features(board)
    return {
        "fen": fen,
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "feature_count": FEATURE_COUNT,
        "white_features": features.white,
        "black_features": features.black,
    }
