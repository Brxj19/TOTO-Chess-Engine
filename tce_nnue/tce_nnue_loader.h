#ifndef TCE_NNUE_LOADER_H
#define TCE_NNUE_LOADER_H

#include "tce_nnue_format.h"

#include <stddef.h>

int tce_nnue_loader_load(const char *path);
void tce_nnue_loader_free(void);

const TceNnueFileHeader *tce_nnue_loader_header(void);
const TceNnueTensorInfo *tce_nnue_loader_tensors(void);
const unsigned char *tce_nnue_loader_payload(void);
size_t tce_nnue_loader_payload_size(void);
const char *tce_nnue_loader_metadata_json(void);
int tce_nnue_loader_is_loaded(void);

#endif
