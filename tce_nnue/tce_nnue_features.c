#include "tce_nnue_features.h"

#include <stdio.h>

int tce_nnue_validate_features(const int *features, int count, int feature_count)
{
    if (!features && count > 0) {
        fprintf(stderr, "tce_nnue: null feature pointer\n");
        return 1;
    }

    if (count < 0) {
        fprintf(stderr, "tce_nnue: negative feature count\n");
        return 1;
    }

    for (int i = 0; i < count; i++) {
        if (features[i] < 0 || features[i] >= feature_count) {
            fprintf(stderr, "tce_nnue: feature id %d outside [0, %d)\n",
                    features[i], feature_count);
            return 1;
        }
    }

    return 0;
}
