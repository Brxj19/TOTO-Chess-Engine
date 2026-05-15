# Opening Suite Testing

The first Cute Chess match repeated very similar games because both engines were deterministic and no opening suite or book was used. With the same start position, time control, and engine settings, the search can keep choosing the same early moves, so a 20-game match may test one narrow line instead of general stability.

Opening suites give each game a different starting position. This is for testing variety, not training.

## Build An EPD Suite

Example from a Lichess PGN:

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

## Cute Chess GUI

Use:

```text
Opening suite -> PGN/EPD file -> data/openings/tce_openings_50.epd
Opening order -> Sequential
Swap sides -> ON
```

`Swap sides` is important because each opening position should be tested with both engines playing both colors.

## Stability Tests

Suggested progression:

```text
10 games  - quick crash/illegal-move smoke test
20 games  - basic stability and repeated-blunder check
50 games  - more useful backend comparison before retraining/exporting again
```

Keep the Stockfish backend as the baseline opponent while measuring the TCE-owned NNUE backend.
