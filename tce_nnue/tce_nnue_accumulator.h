#ifndef TCE_NNUE_ACCUMULATOR_H
#define TCE_NNUE_ACCUMULATOR_H

#include <stdint.h>

int tce_nnue_refresh_accumulator(
    const int16_t *ft_weight,
    int half_dim,
    const int *features,
    int feature_count,
    int32_t *accumulator
);

#endif
