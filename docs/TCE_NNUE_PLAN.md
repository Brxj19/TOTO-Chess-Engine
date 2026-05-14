# TCE-Owned NNUE Implementation Plan

This document describes how TOTO Chess Engine currently evaluates positions with the imported Stockfish NNUE probe, and proposes a plan for adding a TCE-owned NNUE implementation without changing engine behavior in the planning phase.

## 1. Current Evaluation And Stockfish NNUE Integration

TCE currently has one static evaluation entry point in `toto.c`:

- `evaluate()` builds `pieces[33]` and `squares[33]` arrays from the engine's global bitboards.
- TCE piece enums are `P, N, B, R, Q, K, p, n, b, r, q, k`.
- TCE square enums run from `a8` through `h1`.
- Stockfish NNUE expects piece codes `wking=1, wqueen=2, wrook=3, wbishop=4, wknight=5, wpawn=6, bking=7, ... bpawn=12`.
- Stockfish NNUE expects square codes from `A1=0` through `H8=63`.
- `nnue_pieces[12]` maps TCE piece IDs to Stockfish piece IDs.
- `nnue_squares[64]` maps TCE square IDs to Stockfish square IDs.
- The white king is written to `pieces[0]` / `squares[0]`.
- The black king is written to `pieces[1]` / `squares[1]`.
- Other pieces are appended after the kings.
- Both arrays are terminated with `0`.

The current final static evaluation is:

```c
return (evaluate_nnue(side, pieces, squares) * (100 - fifty) / 100);
```

This means the imported NNUE score is already expected to be from the current side-to-move perspective, and TCE applies only the fifty-move rule scaling before search consumes the score.

Search uses `evaluate()` in quiescence and main alpha-beta search, so any replacement NNUE module should preserve this same score contract until there is a deliberate strength-testing phase.

At startup, `init_all()` calls:

```c
init_nnue("nn-eba324f53044.nnue");
```

The UCI loop advertises:

```text
option name EvalFile type string default nn-eba324f53044.nnue
```

The UCI `setoption name EvalFile value ...` parser stores the requested path in `evalFile`, but currently does not reload the network. The code contains a TODO for that behavior.

## 2. Files Currently Handling NNUE

Current NNUE-related files and responsibilities:

- `toto.c`
  - Includes `nnue_eval.h`.
  - Defines the engine version string as `-1.0 + SF NNUE`.
  - Stores `evalFile`.
  - Converts TCE board state to Stockfish probe arrays inside `evaluate()`.
  - Calls `evaluate_nnue(side, pieces, squares)`.
  - Advertises the `EvalFile` UCI option.
  - Initializes the default network from `init_all()`.

- `nnue_eval.h`
  - Declares the C wrapper API:
    - `init_nnue(char *filename)`
    - `evaluate_nnue(int player, int *pieces, int *squares)`
    - `evaluate_fen_nnue(char *fen)`

- `nnue_eval.c`
  - Includes `./nnue/nnue.h`.
  - Implements thin wrappers around the imported probe:
    - `nnue_init()`
    - `nnue_evaluate()`
    - `nnue_evaluate_fen()`

- `nnue/nnue.h`
  - Exposes the imported Stockfish-compatible NNUE probe API.
  - Defines probe-side `Position` and `Accumulator` structures.
  - Documents the required piece and square array format.

- `nnue/nnue.cpp`
  - Contains the imported Stockfish NNUE loader, verifier, feature transformer, accumulator refresh, SIMD-aware network propagation, FEN decoder, and evaluation API.
  - Verifies the current `.nnue` binary against fixed Stockfish metadata and size.
  - Uses a 256-half-dimension transformer and hidden layers of 32 and 32.

- `nnue/misc.h` and `nnue/misc.cpp`
  - Provide file mapping, endian reading, and platform utility code used by `nnue/nnue.cpp`.

- `nn-eba324f53044.nnue`
  - The default imported Stockfish network file.

- `makefile`
  - Builds `toto.c`, `nnue_eval.c`, `nnue/nnue.cpp`, and `nnue/misc.cpp` into the `tce` binary.

There is also a `tce.c` file with duplicated engine code and the same NNUE integration pattern. The build currently uses `toto.c`, so the first production integration should target `toto.c` unless the project later consolidates the duplicate file.

## 3. Where A TCE-Owned NNUE Module Should Connect

The cleanest connection point is the existing wrapper boundary, not the search code.

Recommended first integration:

1. Keep `evaluate()` in `toto.c` as the only search-facing static evaluation entry point.
2. Replace or extend the implementation behind `nnue_eval.h` so `evaluate_nnue()` can call either:
   - the current imported Stockfish probe, or
   - a new TCE-owned NNUE evaluator.
3. Preserve the current return contract:
   - input side is TCE `white`/`black`;
   - return value is centipawns from side-to-move perspective;
   - `evaluate()` remains responsible for applying the fifty-move scaling.

Recommended future cleanup after parity tests:

- Rename the wrapper API to TCE-specific names such as:
  - `tce_nnue_init(const char *path)`
  - `tce_nnue_evaluate_position(const TceNnuePosition *position)`
  - `tce_nnue_evaluate_arrays(int side, const int *pieces, const int *squares)`
- Move Stockfish compatibility array conversion out of `toto.c` and into a TCE-owned adapter.
- Eventually evaluate directly from TCE bitboards instead of constructing Stockfish-style arrays on every static evaluation.

The eventual direct-bitboard API should accept:

- side to move;
- `bitboards[12]`;
- optionally `occupancies[3]`;
- a reusable accumulator object stored per search stack ply or position state.

That direct API would let TCE support incremental accumulator updates in `make_move()` / `take_back()` later, while the first version can safely use full accumulator refresh for correctness.

## 4. Proposed File Structure For `tce_nnue/`

Proposed engine-side layout:

```text
tce_nnue/
├── tce_nnue.h
├── tce_nnue.c
├── tce_nnue_arch.h
├── tce_nnue_features.h
├── tce_nnue_features.c
├── tce_nnue_loader.h
├── tce_nnue_loader.c
├── tce_nnue_accumulator.h
├── tce_nnue_accumulator.c
├── tce_nnue_network.h
├── tce_nnue_network.c
├── tce_nnue_format.h
└── README.md
```

Responsibilities:

- `tce_nnue.h`
  - Public engine API.
  - Stable types for status, score, network handle, and position input.

- `tce_nnue.c`
  - Public API implementation.
  - Owns module initialization, active network state, and high-level evaluation flow.

- `tce_nnue_arch.h`
  - Network architecture constants: feature count, half dimensions, layer sizes, quantization parameters, activation limits, and score scale.

- `tce_nnue_features.h` / `tce_nnue_features.c`
  - TCE feature indexing.
  - Converts TCE pieces and squares into feature IDs.
  - Provides both array-based compatibility input and direct bitboard input.

- `tce_nnue_loader.h` / `tce_nnue_loader.c`
  - Loads and validates `.tcennue` files.
  - Checks magic, version, architecture metadata, payload lengths, and checksum.

- `tce_nnue_accumulator.h` / `tce_nnue_accumulator.c`
  - Full accumulator refresh first.
  - Later incremental add/remove/update APIs for `make_move()` and `take_back()`.

- `tce_nnue_network.h` / `tce_nnue_network.c`
  - Quantized inference implementation.
  - Starts scalar and portable.
  - Later can add SSE2/AVX2 specializations behind compile-time checks.

- `tce_nnue_format.h`
  - On-disk constants and structs for `.tcennue`.

- `README.md`
  - Developer notes for architecture, tests, and format compatibility.

Initial makefile integration can add the C files above while keeping the current Stockfish files compiled until the TCE evaluator is ready to replace them.

## 5. Proposed Python Training Pipeline Under `tools/nnue_train/`

Proposed training layout:

```text
tools/nnue_train/
├── README.md
├── requirements.txt
├── tce_nnue_train/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── fen.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   ├── validate.py
│   ├── quantize.py
│   ├── export_tcennue.py
│   └── selfplay.py
└── configs/
    └── baseline.yaml
```

Pipeline stages:

1. Data collection
   - Accept FEN/score/game datasets.
   - Support self-play data later through `selfplay.py`.
   - Store side to move, result target, optional search score, ply, and game result.

2. Parsing and legality filtering
   - Parse FEN into TCE-compatible piece and square orientation.
   - Reject invalid positions, missing kings, illegal side-to-move values, and positions that cannot map to the selected feature set.

3. Feature extraction
   - Implement exactly the same feature indexing as `tce_nnue_features.c`.
   - Include a test vector file so C and Python feature IDs are byte-for-byte comparable.

4. Model definition
   - Start with a small HalfKP-style architecture close enough to the current runtime shape to reduce integration risk:
     - sparse feature transformer;
     - two perspective accumulators;
     - concatenated side-to-move and opponent perspectives;
     - dense hidden layer 1;
     - dense hidden layer 2;
     - scalar centipawn output.

5. Training
   - Combine value targets from game result and engine/search scores.
   - Use configurable loss mixing, learning rate schedule, batch size, validation split, and checkpoint frequency.

6. Validation
   - Track loss, centipawn error, sign accuracy, tactical position subsets, and simple head-to-head engine matches when available.

7. Quantization
   - Convert trained floating-point weights to fixed-point integer weights.
   - Emit quantization metadata needed by the C inference code.
   - Run Python quantized inference against floating-point inference to measure drift.

8. Export
   - Write `.tcennue` files with a deterministic header, metadata block, tensor directory, quantized tensor payloads, and checksum.
   - Export JSON metadata next to the binary for debugging and reproducibility.

9. C/Python parity tests
   - Keep a small set of fixed FENs.
   - Assert Python quantized inference and C inference produce the same score or stay within a very small tolerance.

## 6. Proposed `.tcennue` Binary Format

Goals:

- TCE-owned and independent of Stockfish `.nnue` file assumptions.
- Easy to validate before loading.
- Forward-compatible through explicit architecture and tensor metadata.
- Deterministic so training exports are reproducible.

All multi-byte numeric fields should be little-endian.

Proposed top-level layout:

```text
TcennueFile
├── Header
├── Metadata JSON bytes
├── Tensor directory
├── Tensor payload bytes
└── Checksum
```

Proposed header fields:

```text
magic[8]          = "TCENNUE\0"
format_version   = uint32
header_size      = uint32
arch_id          = uint32
feature_set_id   = uint32
endianness       = uint32
quant_scheme     = uint32
half_dim         = uint32
hidden1_dim      = uint32
hidden2_dim      = uint32
output_dim       = uint32
feature_count    = uint32
score_scale      = int32
activation_min   = int32
activation_max   = int32
metadata_offset  = uint64
metadata_size    = uint64
tensor_dir_offset= uint64
tensor_count     = uint32
payload_offset   = uint64
payload_size     = uint64
checksum_type    = uint32
checksum_offset  = uint64
reserved         = fixed zero bytes
```

Tensor directory entry:

```text
name[32]
dtype             = uint32
rank              = uint32
dims[4]           = uint32[4]
scale             = int32
zero_point        = int32
offset            = uint64
size              = uint64
alignment         = uint32
```

Initial tensor names:

- `ft_bias`
- `ft_weight`
- `hidden1_bias`
- `hidden1_weight`
- `hidden2_bias`
- `hidden2_weight`
- `output_bias`
- `output_weight`

Initial data types:

- `int16` for feature-transformer biases and weights.
- `int32` for dense layer biases.
- `int8` or `int16` for dense layer weights, selected by `quant_scheme`.

Recommended validation rules:

- Reject files with a wrong magic value.
- Reject unsupported `format_version`, `arch_id`, `feature_set_id`, or `quant_scheme`.
- Reject tensor shapes that do not match `tce_nnue_arch.h`.
- Reject payload offsets that overlap or exceed file size.
- Verify checksum before using the loaded weights.
- Refuse to evaluate if no valid network is loaded.

## 7. Step-By-Step Implementation Plan

1. Add documentation and tests for the current behavior.
   - Record the current Stockfish NNUE integration and score contract.
   - Add fixed-position smoke tests around `evaluate_nnue()` if a test harness is introduced.

2. Add the `tce_nnue/` skeleton behind a compile-time flag.
   - Create public headers and scalar inference stubs.
   - Keep the current Stockfish path as the default.

3. Implement `.tcennue` loader validation.
   - Parse header and tensor directory.
   - Validate architecture and payload bounds.
   - Load tensors into aligned runtime memory.

4. Implement feature indexing in C.
   - Start with the same effective inputs as the current Stockfish-compatible path.
   - Add C test vectors for TCE square orientation and piece mapping.

5. Implement scalar full-refresh inference.
   - Build both perspective accumulators from the full position.
   - Concatenate side-to-move and opponent perspectives.
   - Run quantized dense layers.
   - Return centipawns from side-to-move perspective.

6. Add Python feature extraction and parity tests.
   - Ensure Python and C produce identical feature IDs for fixed FENs.
   - Keep test vectors checked into the repo.

7. Add Python model training and quantized export.
   - Train a baseline network.
   - Export `.tcennue`.
   - Compare Python quantized inference against C scalar inference.

8. Wire TCE-owned NNUE behind UCI/config control.
   - Add a switch such as `option name EvalBackend type combo default stockfish var stockfish var tce`.
   - Make `EvalFile` load the selected backend's file.
   - Preserve the old Stockfish `.nnue` path until TCE-owned strength is proven.

9. Add incremental accumulator support.
   - Introduce per-ply accumulator storage.
   - Update accumulators in `make_move()` / `take_back()` or through a search stack position object.
   - Fall back to full refresh when an incremental state is unavailable.

10. Optimize inference.
    - Add alignment and cache-friendly tensor layout.
    - Add optional SIMD implementations after scalar parity is locked down.
    - Benchmark nodes per second and evaluation latency.

11. Strength test and tune.
    - Run fixed depth and timed matches against the current Stockfish-backed evaluator.
    - Track Elo, crash rate, illegal move rate, and tactical regression suites.
    - Iterate on feature set, training data, and quantization.

12. Retire the imported Stockfish path only after parity and strength goals are met.
    - Remove `nnue/` and `nnue_eval.c` only when the TCE-owned evaluator can fully replace them.
    - Update README, makefile, UCI option docs, and release packaging at the same time.
