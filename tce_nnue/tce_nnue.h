#ifndef TCE_NNUE_H
#define TCE_NNUE_H

int tce_nnue_load(const char *path);
void tce_nnue_free(void);

int tce_nnue_evaluate_sparse(
    const int *white_features,
    int white_count,
    const int *black_features,
    int black_count,
    int side_to_move,
    int *out_cp
);

#endif
