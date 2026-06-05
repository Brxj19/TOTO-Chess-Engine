# TCE NNUE Training Improvement Plan

## Baseline Result

The TCE-owned NNUE backend is integrated and stable enough to play games, but the first match result was poor: TCE-OwnedNNUE lost against TCE-StockfishBackend. There were no crashes or illegal moves, so the integration is healthy, but the trained network is weak.

The main observed weakness is material and tactical blindness. One concrete example was allowing `Qxf6` followed by `Nxf6`, where the model did not punish queen loss strongly enough.

## Why Opening Suites Matter

The Cute Chess match repeated very similar games because no opening suite or book was used. Deterministic engines starting from the same position often choose the same moves, so repeated games do not give enough variety.

Build an opening suite:

```sh
python3 tools/nnue_train/build_opening_suite.py \
  --pgn data/pgn/lichess_db_standard_rated_2013-01.pgn \
  --output data/openings/tce_openings_50.epd \
  --max-games 10000 \
  --min-elo 1600 \
  --min-ply 8 \
  --max-ply 24 \
  --every-n-plies 4 \
  --limit 50 \
  --seed 1
```

In Cute Chess GUI:

```text
Opening suite -> PGN/EPD file -> data/openings/tce_openings_50.epd
Opening order -> Sequential
Swap sides -> ON
```

Run 10 games for smoke, 20 games for a stability check, and 50 games before drawing strength conclusions.

## Tactical Sanity Data

The network needs explicit material/tactical examples: hanging queen, hanging rook, forks, pins, skewers, queen trades, promotion races, pawn endgames, and mate threats.

Generate curated FENs:

```sh
python3 tools/nnue_train/generate_tactical_sanity_fens.py \
  --output data/fens/tactical_material_sanity_v2.fen \
  --metadata-output data/fens/tactical_material_sanity_v2_meta.csv
```

The expanded generator creates a few hundred deterministic tactical/material positions and a metadata CSV with categories such as `queen_up`, `queen_down`, `simple_capture`, `promotion_race`, `queen_trade`, and pawn endgames.

Label them with deeper Stockfish:

```sh
python3 tools/nnue_train/label_positions.py \
  --engine-path stockfish \
  --input data/fens/tactical_material_sanity_v2.fen \
  --output data/labels/tactical_material_sanity_v2_depth14.csv \
  --depth 14 \
  --threads 4 \
  --hash 512 \
  --clamp-cp 3000
```

## Blunder Windows From Matches

Extract positions around big eval swings from TCE's own losing games:

```sh
python3 tools/nnue_train/extract_blunder_windows_from_pgn.py \
  --pgn data/matches/tce_owned_vs_stockfish_20games.pgn \
  --output data/fens/tce_blunder_windows.fen \
  --swing-threshold 500 \
  --window 1
```

Then label:

```sh
python3 tools/nnue_train/label_positions.py \
  --engine-path stockfish \
  --input data/fens/tce_blunder_windows.fen \
  --output data/labels/tce_blunder_windows_depth14.csv \
  --depth 14 \
  --threads 4 \
  --hash 512 \
  --clamp-cp 3000
```

## Merge, Retrain, Export

Merge the base and tactical datasets, oversampling the tactical labels:

```sh
python3 tools/nnue_train/merge_labelled_datasets.py \
  --inputs data/labels/lichess_2013_01.csv \
           data/labels/tactical_material_sanity_v2_depth14.csv \
           data/labels/tce_owned_vs_stockfish_blunder_windows_depth14.csv \
  --oversample data/labels/tactical_material_sanity_v2_depth14.csv:10 \
  --oversample data/labels/tce_owned_vs_stockfish_blunder_windows_depth14.csv:5 \
  --output data/labels/tce_nnue_improved_v2.csv \
  --seed 11 \
  --add-mirrors
```

Use `--add-mirrors` to add color-swapped mirrored positions. This fixes the earlier dataset imbalance where almost all examples were white-to-move.

Build features:

```sh
python3 tools/nnue_train/build_features.py \
  --input data/labels/tce_nnue_improved_v2.csv \
  --output data/nnue/tce_nnue_improved_v2_features.npz \
  --clamp-cp 2000 \
  --keep-duplicates
```

Use `--keep-duplicates` when the merged CSV intentionally oversamples tactical or blunder-window rows. Without it, feature building deduplicates FENs and removes the extra tactical weighting.

Retrain with validation and early stopping:

```sh
python3 tools/nnue_train/train_tce_nnue.py \
  --data data/nnue/tce_nnue_improved_v2_features.npz \
  --output-dir data/nnue_runs/improved_v2 \
  --epochs 10 \
  --batch-size 512 \
  --lr 0.001 \
  --val-ratio 0.1 \
  --target-scale 1000 \
  --half-dim 128 \
  --device auto \
  --seed 11 \
  --num-workers 0 \
  --patience 4 \
  --min-delta 0.0001
```

Export:

```sh
python3 tools/nnue_train/export_tcennue.py \
  --checkpoint data/nnue_runs/improved_v2/checkpoints/best.pt \
  --output data/nnue_runs/improved_v2/tce_improved_v2.tcennue
```

## Tactical Sanity Validation

Validation should include tactical sanity pass rates, not just overall validation MAE:

```sh
python3 tools/nnue_train/validate_tactical_sanity.py \
  --checkpoint data/nnue_runs/improved_v2/checkpoints/best.pt \
  --labels data/labels/tactical_material_sanity_v2_depth14.csv \
  --metadata data/fens/tactical_material_sanity_v2_meta.csv \
  --output data/nnue_runs/improved_v2/tactical_sanity_v2_validation.csv \
  --device auto \
  --report-cap-cp 3000
```

The validator prints target centipawns, predicted centipawns, absolute error, sign correctness, material pass/fail, and per-category summaries. Mate labels are still preserved as raw values, but reporting can cap them so normal material cases remain visible.

## Export Checks

After export, run loader, inference parity, backend regression, and benchmark checks:

```sh
python3 tools/nnue_train/dump_inference_vectors.py \
  --data data/nnue/tce_nnue_improved_v2_features.npz \
  --checkpoint data/nnue_runs/improved_v2/checkpoints/best.pt \
  --tcennue data/nnue_runs/improved_v2/tce_improved_v2.tcennue \
  --output tools/nnue_train/test_vectors/inference_improved_v2_sample.json \
  --samples 16

make check-tcennue FILE=data/nnue_runs/improved_v2/tce_improved_v2.tcennue

make check-tcennue-infer \
  FILE=data/nnue_runs/improved_v2/tce_improved_v2.tcennue \
  VECTORS=tools/nnue_train/test_vectors/inference_improved_v2_sample.json

python3 tools/nnue_train/test_eval_backends.py \
  --engine ./tce \
  --tcennue data/nnue_runs/improved_v2/tce_improved_v2.tcennue \
  --depth 2

python3 tools/nnue_train/benchmark_eval_backends.py \
  --engine ./tce \
  --tcennue data/nnue_runs/improved_v2/tce_improved_v2.tcennue \
  --depth 4 \
  --output data/nnue_runs/improved_v2/backend_benchmark.csv
```

## Cute Chess Match Harness

Rerun Cute Chess with the opening suite and compare against TCE-StockfishBackend and the previous TCE-owned network. If `cutechess-cli` is installed, use:

```sh
GAMES=10 tools/nnue_train/run_cutechess_opening_match.sh
GAMES=20 tools/nnue_train/run_cutechess_opening_match.sh
GAMES=50 tools/nnue_train/run_cutechess_opening_match.sh
```

The script uses `data/openings/tce_openings_50.epd`, sequential opening order, and repeat/swap sides behavior. If using the Cute Chess GUI instead, load `data/openings/tce_openings_50.epd`, set opening order to sequential, and turn swap sides on.
