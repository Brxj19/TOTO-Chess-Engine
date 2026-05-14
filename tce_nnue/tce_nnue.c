#include "tce_nnue.h"

#include "tce_nnue_loader.h"

int tce_nnue_load(const char *path)
{
    return tce_nnue_loader_load(path);
}

void tce_nnue_free(void)
{
    tce_nnue_loader_free();
}
