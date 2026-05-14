#ifndef TCE_NNUE_NETWORK_H
#define TCE_NNUE_NETWORK_H

int tce_nnue_evaluate_sparse(
    const int *white_features,
    int white_count,
    const int *black_features,
    int black_count,
    int side_to_move,
    int *out_cp
);

#endif
