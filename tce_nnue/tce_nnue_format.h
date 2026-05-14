#ifndef TCE_NNUE_FORMAT_H
#define TCE_NNUE_FORMAT_H

#include <stdint.h>

#define TCE_NNUE_MAGIC "TCENNUE"
#define TCE_NNUE_MAGIC_SIZE 8
#define TCE_NNUE_FORMAT_VERSION 1u
#define TCE_NNUE_HEADER_SIZE 52u
#define TCE_NNUE_CHECKSUM_SIZE 32u
#define TCE_NNUE_TENSOR_COUNT 7u

enum {
    TCE_NNUE_TENSOR_FT_WEIGHT,
    TCE_NNUE_TENSOR_HIDDEN1_WEIGHT,
    TCE_NNUE_TENSOR_HIDDEN1_BIAS,
    TCE_NNUE_TENSOR_HIDDEN2_WEIGHT,
    TCE_NNUE_TENSOR_HIDDEN2_BIAS,
    TCE_NNUE_TENSOR_OUTPUT_WEIGHT,
    TCE_NNUE_TENSOR_OUTPUT_BIAS
};

static const char *const TCE_NNUE_TENSOR_NAMES[TCE_NNUE_TENSOR_COUNT] = {
    "ft_weight",
    "hidden1_weight",
    "hidden1_bias",
    "hidden2_weight",
    "hidden2_bias",
    "output_weight",
    "output_bias"
};

typedef struct {
    char magic[TCE_NNUE_MAGIC_SIZE];
    uint32_t format_version;
    uint32_t feature_count;
    uint32_t half_dim;
    uint32_t hidden1_dim;
    uint32_t hidden2_dim;
    uint32_t output_dim;
    double target_scale;
    uint32_t tensor_count;
    uint64_t metadata_size;
} TceNnueFileHeader;

typedef struct {
    char name[32];
    char dtype[16];
    uint32_t rank;
    uint32_t shape[4];
    uint64_t offset;
    uint64_t size;
} TceNnueTensorInfo;

#endif
