# TCE NNUE Loader

This directory contains the first C-side support for TCE-owned `.tcennue` files.

Current scope:

- read a `.tcennue` file;
- verify the magic bytes and supported format version;
- read the JSON metadata block;
- validate the expected tensor names, dtypes, offsets, and sizes;
- verify the trailing SHA256 checksum;
- expose `tce_nnue_load()` and `tce_nnue_free()`.
- run standalone scalar sparse inference with `tce_nnue_evaluate_sparse()`.

This is still standalone infrastructure. The loaded network is not used by engine evaluation yet. TCE still uses the existing Stockfish NNUE integration, and neither `init_all()` nor `evaluate()` calls this loader or inference path.

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

The next stage is engine-side integration planning after scalar inference parity is stable. That later stage should decide how to pass TCE board state into this module without changing search semantics accidentally.
