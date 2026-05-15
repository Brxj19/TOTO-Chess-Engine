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
python3 tools/nnue_train/generate_tactical_sanity_fens.py
```

Label them with deeper Stockfish:

```sh
python3 tools/nnue_train/label_positions.py \
  --engine-path stockfish \
  --input data/fens/tactical_material_sanity.fen \
  --output data/labels/tactical_material_sanity_depth14.csv \
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
           data/labels/tactical_material_sanity_depth14.csv \
           data/labels/tce_blunder_windows_depth14.csv \
  --oversample data/labels/tactical_material_sanity_depth14.csv:5 \
  --oversample data/labels/tce_blunder_windows_depth14.csv:5 \
  --output data/labels/tce_nnue_improved.csv \
  --seed 1
```

Build features:

```sh
python3 tools/nnue_train/build_features.py \
  --input data/labels/tce_nnue_improved.csv \
  --output data/nnue/tce_nnue_improved_features.npz \
  --clamp-cp 3000
```

Retrain with validation and early stopping:

```sh
python3 tools/nnue_train/train_tce_nnue.py \
  --data data/nnue/tce_nnue_improved_features.npz \
  --output-dir data/nnue_runs/tce_improved \
  --epochs 50 \
  --batch-size 256 \
  --lr 0.001 \
  --val-ratio 0.1 \
  --target-scale 1000 \
  --patience 5 \
  --min-delta 0.0001
```

Export:

```sh
python3 tools/nnue_train/export_tcennue.py \
  --checkpoint data/nnue_runs/tce_improved/checkpoints/best.pt \
  --output data/nnue_runs/tce_improved/tce_improved.tcennue
```

Rerun Cute Chess with the opening suite and compare against TCE-StockfishBackend.
