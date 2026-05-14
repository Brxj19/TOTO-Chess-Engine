#include <stdio.h>

#include "tce_nnue/tce_nnue.h"

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <network.tcennue>\n", argv[0]);
        return 2;
    }

    if (tce_nnue_load(argv[1]) != 0) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        tce_nnue_free();
        return 1;
    }

    printf("loaded successfully\n");
    tce_nnue_free();
    return 0;
}
