# PGN Dataset Preparation

This stage prepares plain FEN text files for future TCE NNUE training. It does not train a network.

## Install Python Dependencies

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/nnue_train/requirements.txt
```

## Download Lichess PGN Data

Lichess publishes monthly rated-game PGN dumps at:

```text
https://database.lichess.org/
```

For a first small experiment, choose one monthly standard chess dump from the "Standard Chess" section. The files are compressed as `.zst`.

Example:

```sh
mkdir -p data/lichess
cd data/lichess
curl -O https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
```

Decompress it before running the extractor:

```sh
zstd -d lichess_db_standard_rated_2013-01.pgn.zst
```

This creates:

```text
lichess_db_standard_rated_2013-01.pgn
```

## Extract FEN Positions

Run the extractor from the repository root:

```sh
python tools/nnue_train/extract_fens_from_pgn.py \
  --pgn data/lichess/lichess_db_standard_rated_2013-01.pgn \
  --output data/nnue/fens.txt \
  --max-games 10000 \
  --skip-plies 12 \
  --every-n-plies 4 \
  --min-elo 1800
```

The output file contains one FEN per line.

## Extractor Behavior

`extract_fens_from_pgn.py`:

- reads PGN games with `python-chess`;
- keeps only games where both `WhiteElo` and `BlackElo` are at least `--min-elo`;
- skips the first `--skip-plies` plies of each accepted game;
- samples one position every `--every-n-plies` plies after that;
- skips illegal move sequences;
- skips checkmate and stalemate positions;
- removes duplicate positions;
- writes one legal FEN per line;
- displays progress with `tqdm`.

Duplicates are detected using board placement, side to move, castling rights, and en-passant square. Halfmove and fullmove counters are not part of the duplicate key because they are not static NNUE evaluation features.

## Label FEN Positions With Stockfish

After extracting FENs, label them with a UCI engine. Stockfish is the first supported engine, but the labelling script uses the generic `python-chess` `SimpleEngine` interface so TOTO can be used later when it exposes the needed UCI evaluation behavior.

Example:

```sh
python tools/nnue_train/label_positions.py \
  --engine-path stockfish \
  --input data/fens/lichess_2013_01.fen \
  --output data/labels/lichess_2013_01_depth8.csv \
  --depth 8 \
  --threads 4 \
  --hash 512 \
  --clamp-cp 3000
```

The output CSV contains:

```text
fen,eval_cp,best_move,depth
```

`eval_cp` is stored from the side-to-move perspective. Normal centipawn values are clamped to `+/- --clamp-cp`. Mate scores are converted into large centipawn targets so they can be stored in the same column:

```text
mate in positive N = +100000 - N
mate in negative N = -100000 + N
```

Use `--resume` to continue an interrupted labelling run. When the output CSV already exists, the script reads existing `fen` rows and skips those positions.

## Build Sparse NNUE Features

After labelling positions, convert the labelled CSV into sparse NNUE feature IDs. This stage still does not train the network. It only prepares compact arrays that a future PyTorch or NumPy training step can load.

Example:

```sh
python3 tools/nnue_train/build_features.py \
  --input data/labels/lichess_2013_01.csv \
  --output data/nnue/lichess_2013_01_features.npz \
  --max-positions 10000 \
  --clamp-cp 2000
```

The first TCE feature set uses:

```text
64 king squares * 12 piece types * 64 piece squares = 49152 feature IDs
```

The builder stores separate sparse feature lists for white and black perspectives. For black perspective, squares are rotated 180 degrees and colors are interpreted relative to the black side, so the feature space remains consistent between both sides. Kings select the feature bucket, but are not emitted as normal piece features in this first dataset format.

The `.npz` file contains flattened feature arrays plus offsets:

```text
fens
white_features
white_offsets
black_features
black_offsets
eval_cp
side_to_move
best_move
depth
feature_count
schema_version
```

For debugging one position:

```sh
python3 tools/nnue_train/build_features.py \
  --debug-fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

To refresh the checked-in sample vectors:

```sh
python3 tools/nnue_train/build_features.py --write-sample-vectors
```

## Train Baseline NNUE

After building sparse features, run the baseline PyTorch trainer. This stage trains a simple NNUE-like model only. It does not quantize the network, export `.tcennue`, or modify the C/C++ engine.

Smoke test:

```sh
python3 tools/nnue_train/train_nnue.py \
  --data data/nnue/lichess_2013_01_features.npz \
  --output-dir data/nnue_runs/smoke \
  --epochs 1 \
  --batch-size 256 \
  --lr 0.001 \
  --val-split 0.1 \
  --target-scale 1000
```

The trainer reads `eval_cp` from the feature `.npz`, normalizes it by `--target-scale`, clamps normalized targets to `[-2.0, 2.0]`, and reports validation MAE in centipawns.

Training output:

```text
checkpoints/best.pt
checkpoints/last.pt
metrics.csv
config.json
```

These generated run directories should stay under `data/` and should not be committed.

## Quantize And Export `.tcennue`

After training a baseline checkpoint, quantize the PyTorch weights and export a deterministic TCE-owned `.tcennue` binary. The exported file is not integrated into engine evaluation yet, and TCE continues using the current Stockfish NNUE path.

Export:

```sh
python3 tools/nnue_train/export_tcennue.py \
  --checkpoint data/nnue_runs/baseline/checkpoints/best.pt \
  --output data/nnue_runs/baseline/tce_baseline.tcennue
```

Inspect:

```sh
python3 tools/nnue_train/export_tcennue.py \
  --inspect data/nnue_runs/baseline/tce_baseline.tcennue
```

The exporter stores:

```text
TCENNUE\0 magic bytes
fixed little-endian header
canonical JSON metadata
tensor payloads in deterministic order
trailing SHA256 checksum
```

Tensor order:

```text
ft_weight
hidden1_weight
hidden1_bias
hidden2_weight
hidden2_bias
output_weight
output_bias
```

Weights are quantized to `int16`; biases are stored as `int32`. Generated `.tcennue` files should stay under `data/` and should not be committed.

## Standalone C Scalar Inference

The C-side loader and scalar inference path can be tested without changing engine behavior. This standalone path loads `.tcennue`, consumes sparse feature IDs, and compares C predictions against Python-generated quantized test vectors.

Dump inference vectors:

```sh
python3 tools/nnue_train/dump_inference_vectors.py \
  --data data/nnue/lichess_2013_01_features.npz \
  --checkpoint data/nnue_runs/baseline/checkpoints/best.pt \
  --tcennue data/nnue_runs/baseline/tce_baseline.tcennue \
  --output tools/nnue_train/test_vectors/inference_sample.json \
  --samples 16
```

Run C parity check:

```sh
make check-tcennue-infer \
  FILE=data/nnue_runs/baseline/tce_baseline.tcennue \
  VECTORS=tools/nnue_train/test_vectors/inference_sample.json
```

This stage still does not call the new NNUE from `evaluate()`, does not modify search, and does not remove the Stockfish NNUE integration.

## CLI Arguments

```text
--pgn              Input PGN file.
--output           Output text file for extracted FENs.
--max-games        Maximum number of Elo-accepted games to process.
--skip-plies       Number of plies to skip from the start of each game.
--every-n-plies    Extract one position every N plies after skipped plies.
--min-elo          Minimum Elo required for both players.
```

## Practical Notes

Start with a small `--max-games` value to validate the pipeline quickly. Larger Lichess dumps can be very large after decompression, so keep them under `data/` and avoid committing generated PGN or FEN files.
