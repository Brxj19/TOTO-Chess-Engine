#include "tce_nnue_accumulator.h"

#include <string.h>

int tce_nnue_refresh_accumulator(
    const int16_t *ft_weight,
    int half_dim,
    const int *features,
    int feature_count,
    int32_t *accumulator
)
{
    if (!ft_weight || !accumulator || half_dim <= 0 || feature_count < 0)
        return 1;

    memset(accumulator, 0, (size_t)half_dim * sizeof(accumulator[0]));

    for (int i = 0; i < feature_count; i++) {
        const int16_t *row = ft_weight + ((size_t)features[i] * (size_t)half_dim);
        for (int j = 0; j < half_dim; j++)
            accumulator[j] += row[j];
    }

    return 0;
}
