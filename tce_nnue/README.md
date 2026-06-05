# TCE NNUE Loader

This directory contains the C-side support for TCE-owned `.tcennue` files.

Current scope:

- read a `.tcennue` file;
- verify the magic bytes and supported format version;
- read the JSON metadata block;
- validate the expected tensor names, dtypes, offsets, and sizes;
- verify the trailing SHA256 checksum;
- expose `tce_nnue_load()` and `tce_nnue_free()`.
- run scalar sparse inference with `tce_nnue_evaluate_sparse()`;
- support the optional engine evaluation backend selected by UCI `EvalBackend`.

The TCE-owned backend is integrated through `evaluate()`, but the Stockfish NNUE path is still present and remains the safe fallback. The UCI backend selector currently defaults to `tce`; a valid `.tcennue` must still be loaded through `EvalFile`, otherwise evaluation falls back to the Stockfish NNUE evaluator instead of crashing.

Current parity workflow:

```sh
python3 tools/nnue_train/dump_inference_vectors.py \
  --data data/nnue/lichess_2013_01_features.npz \
  --checkpoint data/nnue_runs/baseline/checkpoints/best.pt \
  --tcennue data/nnue_runs/baseline/tce_baseline.tcennue \
  --output tools/nnue_train/test_vectors/inference_sample.json \
  --samples 16

make check-tcennue-infer \
  FILE=data/nnue_runs/baseline/tce_baseline.tcennue \
  VECTORS=tools/nnue_train/test_vectors/inference_sample.json
```

The next runtime stage is performance work after strength validation: keep scalar parity stable, avoid hot-path allocation, then consider incremental accumulators and SIMD.
