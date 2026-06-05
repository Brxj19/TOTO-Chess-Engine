#!/usr/bin/env bash
set -euo pipefail

ENGINE="${ENGINE:-./tce}"
TCENNUE="${TCENNUE:-data/nnue_runs/improved_v2/tce_improved_v2.tcennue}"
BASELINE_TCENNUE="${BASELINE_TCENNUE:-data/nnue_runs/improved_tactical/tce_improved_tactical.tcennue}"
OPENINGS="${OPENINGS:-data/openings/tce_openings_50.epd}"
GAMES="${GAMES:-10}"
CONCURRENCY="${CONCURRENCY:-1}"
TIME_CONTROL="${TIME_CONTROL:-40/10}"
OUTPUT="${OUTPUT:-data/matches/tce_improved_v2_opening_suite_${GAMES}games.pgn}"

if ! command -v cutechess-cli >/dev/null 2>&1; then
  echo "cutechess-cli was not found in PATH." >&2
  echo "Install Cute Chess CLI or run this match from the Cute Chess GUI." >&2
  exit 127
fi

mkdir -p "$(dirname "$OUTPUT")"

cutechess-cli \
  -engine name=TCE-StockfishBackend cmd="$ENGINE" option.EvalBackend=stockfish \
  -engine name=TCE-OwnedNNUE-v2 cmd="$ENGINE" option.EvalBackend=tce option.EvalFile="$TCENNUE" \
  -engine name=TCE-OwnedNNUE-previous cmd="$ENGINE" option.EvalBackend=tce option.EvalFile="$BASELINE_TCENNUE" \
  -each proto=uci tc="$TIME_CONTROL" timemargin=500 \
  -games "$GAMES" \
  -rounds 1 \
  -repeat \
  -concurrency "$CONCURRENCY" \
  -openings file="$OPENINGS" format=epd order=sequential plies=0 \
  -pgnout "$OUTPUT" \
  -recover \
  -ratinginterval 10

echo "Wrote match PGN to $OUTPUT"
